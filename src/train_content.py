"""
Content-Based Filtering using TF-IDF
"""
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

def load_movie_data():
    """Load movie metadata"""
    data_path = Path('data/ml-100k/u.item')
    
    movies = pd.read_csv(
        data_path,
        sep='|',
        encoding='latin-1',
        names=['item_id', 'title', 'release_date', 'video_release_date',
               'imdb_url', 'unknown', 'Action', 'Adventure', 'Animation',
               'Children', 'Comedy', 'Crime', 'Documentary', 'Drama',
               'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
               'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    )
    
    return movies

def create_content_features(movies_df):
    """
    Create content-based features from movie metadata
    
    Args:
        movies_df: DataFrame with movie information
    
    Returns:
        Feature matrix and movie IDs
    """
    # Genre columns
    genre_cols = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy',
                  'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
                  'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                  'Thriller', 'War', 'Western']
    
    # Create genre string for each movie
    def get_genres(row):
        genres = [col for col in genre_cols if row[col] == 1]
        return ' '.join(genres)
    
    movies_df['genre_string'] = movies_df.apply(get_genres, axis=1)
    
    # Extract year from title (format: "Movie Title (Year)")
    movies_df['year'] = movies_df['title'].str.extract(r'\((\d{4})\)')
    movies_df['year'] = pd.to_numeric(movies_df['year'], errors='coerce')
    
    # Add year decade as feature
    movies_df['decade'] = (movies_df['year'] // 10 * 10).fillna(0).astype(int)
    movies_df['decade_str'] = movies_df['decade'].apply(lambda x: f"decade_{x}" if x > 0 else "")
    
    # Combine features
    movies_df['content_features'] = (
        movies_df['genre_string'] + ' ' + 
        movies_df['decade_str']
    )
    
    return movies_df

def build_content_model(movies_df):
    """
    Build TF-IDF content-based model
    
    Args:
        movies_df: DataFrame with movie features
    
    Returns:
        TF-IDF vectorizer, feature matrix, item IDs
    """
    # Create TF-IDF features
    tfidf = TfidfVectorizer(
        max_features=100,
        stop_words='english',
        ngram_range=(1, 2)
    )
    
    tfidf_matrix = tfidf.fit_transform(movies_df['content_features'])
    
    print(f"✓ Content features created: {tfidf_matrix.shape}")
    
    return tfidf, tfidf_matrix, movies_df['item_id'].values

def get_similar_items(item_id, tfidf_matrix, item_ids, k=20):
    """
    Find similar items based on content
    
    Args:
        item_id: Target item ID
        tfidf_matrix: TF-IDF feature matrix
        item_ids: Array of item IDs
        k: Number of similar items to return
    
    Returns:
        List of (item_id, similarity_score) tuples
    """
    try:
        # Find index of target item
        idx = np.where(item_ids == item_id)[0][0]
    except IndexError:
        return []
    
    # Compute similarities
    item_vector = tfidf_matrix[idx:idx+1]
    similarities = cosine_similarity(item_vector, tfidf_matrix)[0]
    
    # Get top K (excluding self)
    similar_indices = np.argsort(similarities)[::-1][1:k+1]
    
    results = [
        (item_ids[i], similarities[i]) 
        for i in similar_indices
    ]
    
    return results

def recommend_for_cold_start(user_ratings, tfidf_matrix, item_ids, k=10):
    """
    Generate recommendations for cold-start users
    
    Args:
        user_ratings: List of (item_id, rating) tuples from user
        tfidf_matrix: TF-IDF feature matrix
        item_ids: Array of item IDs
        k: Number of recommendations
    
    Returns:
        List of recommended item IDs with scores
    """
    if not user_ratings:
        # No ratings at all - return popular items
        return []
    
    # Get items similar to what user liked (rating >= 4)
    liked_items = [item_id for item_id, rating in user_ratings if rating >= 4]
    
    if not liked_items:
        # User hasn't liked anything - use all their ratings
        liked_items = [item_id for item_id, rating in user_ratings]
    
    # Aggregate similarity scores
    all_scores = {}
    
    for item_id in liked_items:
        similar = get_similar_items(item_id, tfidf_matrix, item_ids, k=50)
        
        for sim_item, sim_score in similar:
            if sim_item not in [i for i, r in user_ratings]:  # Not already rated
                all_scores[sim_item] = all_scores.get(sim_item, 0) + sim_score
    
    # Sort by score
    recommendations = sorted(
        all_scores.items(), 
        key=lambda x: x[1], 
        reverse=True
    )[:k]
    
    return recommendations

def save_content_model(tfidf, tfidf_matrix, item_ids, filepath='models/content_model.pkl'):
    """Save content-based model"""
    Path('models').mkdir(exist_ok=True)
    
    model_data = {
        'tfidf': tfidf,
        'tfidf_matrix': tfidf_matrix,
        'item_ids': item_ids
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"✓ Content model saved to {filepath}")

def main():
    """Main training pipeline"""
    print("=" * 60)
    print("CONTENT-BASED FILTERING TRAINING")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading movie data...")
    movies = load_movie_data()
    print(f"   Loaded {len(movies):,} movies")
    
    # Create features
    print("\n2. Creating content features...")
    movies = create_content_features(movies)
    
    # Build model
    print("\n3. Building TF-IDF model...")
    tfidf, tfidf_matrix, item_ids = build_content_model(movies)
    
    # Test: Find movies similar to Star Wars
    print("\n4. Testing model...")
    star_wars_id = movies[movies['title'].str.contains('Star Wars', case=False, na=False)]['item_id'].values
    if len(star_wars_id) > 0:
        similar = get_similar_items(star_wars_id[0], tfidf_matrix, item_ids, k=5)
        print(f"\n   Movies similar to Star Wars:")
        for item_id, score in similar:
            title = movies[movies['item_id'] == item_id]['title'].values[0]
            print(f"   - {title} (similarity: {score:.3f})")
    
    # Save
    print("\n5. Saving model...")
    save_content_model(tfidf, tfidf_matrix, item_ids)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)

if __name__ == "__main__":
    main()
