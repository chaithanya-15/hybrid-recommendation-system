"""
FastAPI Backend for Hybrid Recommendation System
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import pickle
import pandas as pd
import numpy as np
import random
import time
from datetime import datetime
from pathlib import Path
from src.train_hybrid import HybridRecommender

# Initialize FastAPI app
app = FastAPI(
    title="Hybrid Recommendation API",
    description="Production-ready movie recommendation system",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models and data at startup
models = {}
movies_df = None

@app.on_event("startup")
async def load_models():
    """Load models when API starts"""
    global models, movies_df
    
    print("Loading models...")
    
    # Load SVD model
    with open('models/svd_model.pkl', 'rb') as f:
        models['svd'] = pickle.load(f)
    
    # Load content model
    with open('models/content_model.pkl', 'rb') as f:
        models['content'] = pickle.load(f)
    
    # Load hybrid model data and initialize recommender
    with open('models/hybrid_model.pkl', 'rb') as f:
        hybrid_data = pickle.load(f)
        
    try:
        models['hybrid'] = HybridRecommender(
            svd_model=hybrid_data['svd_model'],
            content_model=hybrid_data['content_model'],
            item_rating_counts=hybrid_data.get('item_rating_counts', {}),
            cold_start_threshold=hybrid_data.get('cold_start_threshold', 5)
        )
    except Exception as e:
        print(f"Warning: Could not initialize HybridRecommender directly: {e}")
        models['hybrid'] = None
    
    # Load movie metadata
    movies_df = pd.read_csv(
        'data/ml-100k/u.item',
        sep='|',
        encoding='latin-1',
        names=['item_id', 'title', 'release_date', 'video_release_date',
               'imdb_url', 'unknown', 'Action', 'Adventure', 'Animation',
               'Children', 'Comedy', 'Crime', 'Documentary', 'Drama',
               'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
               'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    )
    
    print("✓ Models loaded successfully!")


# Request/Response models
class RecommendationRequest(BaseModel):
    user_id: int
    k: int = 10
    ab_test: bool = False

class RecommendationItem(BaseModel):
    item_id: int
    title: str
    predicted_rating: float
    confidence: float
    genres: List[str]
    is_cold_start_item: bool = False

class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: List[RecommendationItem]
    model_version: str
    inference_time_ms: float
    timestamp: str


# API endpoints
@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "message": "Hybrid Recommendation API v1.0",
        "endpoints": {
            "recommend": "/recommend",
            "docs": "/docs",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "models_loaded": len(models) > 0,
        "movies_loaded": movies_df is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """
    Get personalized movie recommendations
    
    Args:
        request: RecommendationRequest with user_id, k, ab_test
    
    Returns:
        RecommendationResponse with top-K recommendations
    """
    start_time = time.time()
    
    try:
        user_id = request.user_id
        k = min(request.k, 50)  # Cap at 50 recommendations
        
        # A/B testing: randomly choose model
        if request.ab_test:
            model_version = random.choice(['A', 'B'])
            # In production, model A and B would be different versions
            # For now, we'll use the same hybrid model
        else:
            model_version = 'hybrid_v1'
        
        # Get user's past ratings to determine cold-start status
        ratings_df = pd.read_csv(
            'data/ml-100k/u.data',
            sep='\t',
            names=['user_id', 'item_id', 'rating', 'timestamp']
        )
        
        user_ratings = ratings_df[ratings_df['user_id'] == user_id]
        user_ratings_list = list(zip(user_ratings['item_id'], user_ratings['rating']))
        
        # Get recommendations using hybrid model (which handles item cold-start internally)
        hybrid_model = models.get('hybrid')
        
        if hybrid_model:
            top_k = hybrid_model.recommend(user_id, user_ratings_list, n=k)
        else:
            # Fallback if hybrid fails
            rated_items = set(user_ratings['item_id'].values)
            all_items = movies_df['item_id'].values
            unrated_items = [item for item in all_items if item not in rated_items]
            
            svd_model = models['svd']
            predictions = []
            for item_id in unrated_items:
                pred = svd_model.predict(user_id, item_id)
                predictions.append((item_id, pred.est, False))
            predictions.sort(key=lambda x: x[1], reverse=True)
            top_k = predictions[:k]
        
        # Format response
        recommendations = []
        for item_id, pred_rating, is_cold_start_item in top_k:
            movie = movies_df[movies_df['item_id'] == item_id].iloc[0]
            
            # Get genres
            genre_cols = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy',
                          'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
                          'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                          'Thriller', 'War', 'Western']
            genres = [col for col in genre_cols if movie[col] == 1]
            
            # Calculate confidence (simplified)
            confidence = min(pred_rating / 5.0, 1.0)
            
            recommendations.append(
                RecommendationItem(
                    item_id=int(item_id),
                    title=movie['title'],
                    predicted_rating=round(float(pred_rating), 2),
                    confidence=round(confidence, 2),
                    genres=genres,
                    is_cold_start_item=is_cold_start_item
                )
            )
        
        # Calculate inference time
        inference_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Log request (in production, log to database)
        print(f"[{datetime.now()}] user={user_id}, k={k}, "
              f"latency={inference_time:.2f}ms")
        
        return RecommendationResponse(
            user_id=user_id,
            recommendations=recommendations,
            model_version=model_version,
            inference_time_ms=round(inference_time, 2),
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/movie/{item_id}")
async def get_movie_details(item_id: int):
    """Get details for a specific movie"""
    try:
        movie = movies_df[movies_df['item_id'] == item_id].iloc[0]
        
        genre_cols = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy',
                      'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
                      'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                      'Thriller', 'War', 'Western']
        genres = [col for col in genre_cols if movie[col] == 1]
        
        return {
            "item_id": int(item_id),
            "title": movie['title'],
            "release_date": str(movie['release_date']),
            "imdb_url": movie['imdb_url'],
            "genres": genres
        }
    except IndexError:
        raise HTTPException(status_code=404, detail="Movie not found")


@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    ratings_df = pd.read_csv(
        'data/ml-100k/u.data',
        sep='\t',
        names=['user_id', 'item_id', 'rating', 'timestamp']
    )
    
    return {
        "total_users": int(ratings_df['user_id'].nunique()),
        "total_movies": int(ratings_df['item_id'].nunique()),
        "total_ratings": int(len(ratings_df)),
        "sparsity": float(1 - (len(ratings_df) / (ratings_df['user_id'].nunique() * ratings_df['item_id'].nunique()))),
        "avg_rating": float(ratings_df['rating'].mean())
    }


# Run with: uvicorn api:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
