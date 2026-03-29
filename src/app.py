"""
Streamlit Frontend for Movie Recommendation System
"""
import streamlit as st
import requests
import pandas as pd
import pickle
from train_hybrid import HybridRecommender

# Page config
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# Load data
@st.cache_data
def load_data():
    """Load movie and ratings data"""
    movies = pd.read_csv(
        'data/ml-100k/u.item',
        sep='|',
        encoding='latin-1',
        names=['item_id', 'title', 'release_date', 'video_release_date',
               'imdb_url', 'unknown', 'Action', 'Adventure', 'Animation',
               'Children', 'Comedy', 'Crime', 'Documentary', 'Drama',
               'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
               'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    )
    
    ratings = pd.read_csv(
        'data/ml-100k/u.data',
        sep='\t',
        names=['user_id', 'item_id', 'rating', 'timestamp']
    )
    
    return movies, ratings

@st.cache_resource
def load_model():
    """Load trained hybrid model"""
    with open('models/hybrid_model.pkl', 'rb') as f:
        hybrid_data = pickle.load(f)
        
    hybrid_model = HybridRecommender(
        svd_model=hybrid_data['svd_model'],
        content_model=hybrid_data['content_model'],
        item_rating_counts=hybrid_data.get('item_rating_counts', {}),
        cold_start_threshold=hybrid_data.get('cold_start_threshold', 5)
    )
    return hybrid_model

# Load data
movies_df, ratings_df = load_data()
model = load_model()

# Title
st.title("🎬 Hybrid Movie Recommendation System")
st.markdown("*Powered by Collaborative + Content-Based Filtering*")

# Sidebar
st.sidebar.header("Settings")
user_id = st.sidebar.selectbox(
    "Select User ID",
    options=sorted(ratings_df['user_id'].unique()),
    index=0
)

num_recommendations = st.sidebar.slider(
    "Number of Recommendations",
    min_value=5,
    max_value=20,
    value=10
)

# Main content
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("User Profile")
    
    # Get user stats
    user_ratings = ratings_df[ratings_df['user_id'] == user_id]
    
    st.metric("Total Ratings", len(user_ratings))
    st.metric("Average Rating", f"{user_ratings['rating'].mean():.2f}")
    
    st.success("✓ Rating history loaded")
    st.info("System dynamically uses Content-Based Filtering for niche movies (❄️).")
    
    # Show user's top-rated movies
    st.subheader("Your Top Rated Movies")
    top_rated = user_ratings.sort_values('rating', ascending=False).head(5)
    top_rated = top_rated.merge(movies_df[['item_id', 'title']], on='item_id')
    
    for _, row in top_rated.iterrows():
        st.write(f"⭐ **{row['rating']}** - {row['title']}")

with col2:
    st.subheader(f"Top {num_recommendations} Recommendations")
    
    if st.button("Get Recommendations", type="primary"):
        with st.spinner("Generating recommendations..."):
            user_ratings_list = list(zip(user_ratings['item_id'], user_ratings['rating']))
            
            top_k = model.recommend(user_id, user_ratings_list, n=num_recommendations)
            
            # Display recommendations
            for idx, (item_id, pred_rating, is_cold_start_item) in enumerate(top_k, 1):
                movie = movies_df[movies_df['item_id'] == item_id].iloc[0]
                
                # Get genres
                genre_cols = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy',
                              'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
                              'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                              'Thriller', 'War', 'Western']
                genres = [col for col in genre_cols if movie[col] == 1]
                
                # Create card
                with st.container():
                    if is_cold_start_item:
                        st.markdown(f"### {idx}. {movie['title']} ❄️ *Niche Pick*")
                    else:
                        st.markdown(f"### {idx}. {movie['title']}")
                    
                    st.markdown(f"**Predicted Rating:** ⭐ {pred_rating:.2f} / 5.0")
                    st.markdown(f"**Genres:** {', '.join(genres)}")
                    
                    # Add to watchlist button
                    if st.button(f"Add to Watchlist", key=f"add_{item_id}"):
                        st.success(f"Added {movie['title']} to watchlist!")
                    
                    st.divider()

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    """
    This recommendation system uses:
    - **Collaborative Filtering (SVD)** for popular items
    - **Content-Based Filtering (TF-IDF)** for cold-start items (< 5 ratings)
    - **Hybrid Approach** for best results
    
    **Dataset:** MovieLens 100K
    """
)

# Statistics
st.sidebar.markdown("### Dataset Statistics")
st.sidebar.metric("Total Users", ratings_df['user_id'].nunique())
st.sidebar.metric("Total Movies", ratings_df['item_id'].nunique())
st.sidebar.metric("Total Ratings", len(ratings_df))
sparsity = 1 - (len(ratings_df) / (ratings_df['user_id'].nunique() * ratings_df['item_id'].nunique()))
st.sidebar.metric("Data Sparsity", f"{sparsity*100:.1f}%")
