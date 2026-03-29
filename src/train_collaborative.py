"""
Collaborative Filtering Model using SVD
"""
import pickle
import numpy as np
from surprise import SVD, Dataset, Reader
from surprise.model_selection import GridSearchCV, train_test_split
from pathlib import Path
import pandas as pd

def load_data():
    """Load MovieLens data"""
    data_path = Path('data/ml-100k/u.data')
    
    # Load ratings
    ratings = pd.read_csv(
        data_path,
        sep='\t',
        names=['user_id', 'item_id', 'rating', 'timestamp']
    )
    
    return ratings

def train_svd_model(ratings_df, tune_hyperparameters=False):
    """
    Train SVD collaborative filtering model
    
    Args:
        ratings_df: DataFrame with user_id, item_id, rating columns
        tune_hyperparameters: Whether to run grid search
    
    Returns:
        Trained SVD model
    """
    # Create Surprise dataset
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(
        ratings_df[['user_id', 'item_id', 'rating']], 
        reader
    )
    
    if tune_hyperparameters:
        print("Running hyperparameter tuning...")
        param_grid = {
            'n_factors': [50, 100, 150],
            'n_epochs': [10, 20, 30],
            'lr_all': [0.002, 0.005, 0.01],
            'reg_all': [0.02, 0.05, 0.1]
        }
        
        gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3, n_jobs=-1)
        gs.fit(data)
        
        print(f"Best RMSE: {gs.best_score['rmse']:.4f}")
        print(f"Best params: {gs.best_params['rmse']}")
        
        # Use best parameters
        best_params = gs.best_params['rmse']
        algo = SVD(**best_params)
    else:
        # Use pre-tuned parameters
        algo = SVD(
            n_factors=100,
            n_epochs=20,
            lr_all=0.005,
            reg_all=0.02,
            random_state=42
        )
    
    # Train on full dataset
    trainset = data.build_full_trainset()
    algo.fit(trainset)
    
    print("✓ SVD model trained successfully")
    return algo

def evaluate_model(algo, ratings_df, test_size=0.2):
    """Evaluate model on test set"""
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(
        ratings_df[['user_id', 'item_id', 'rating']], 
        reader
    )
    
    trainset, testset = train_test_split(data, test_size=test_size, random_state=42)
    
    # Train
    algo.fit(trainset)
    
    # Test
    predictions = algo.test(testset)
    
    # Calculate metrics
    errors = [abs(pred.est - pred.r_ui) for pred in predictions]
    mae = np.mean(errors)
    rmse = np.sqrt(np.mean([e**2 for e in errors]))
    
    print(f"\nTest Set Performance:")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    
    return predictions

def save_model(model, filepath='models/svd_model.pkl'):
    """Save trained model"""
    Path('models').mkdir(exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"✓ Model saved to {filepath}")

def main():
    """Main training pipeline"""
    print("=" * 60)
    print("COLLABORATIVE FILTERING TRAINING")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading data...")
    ratings = load_data()
    print(f"   Loaded {len(ratings):,} ratings")
    print(f"   Users: {ratings['user_id'].nunique():,}")
    print(f"   Items: {ratings['item_id'].nunique():,}")
    
    # Train model
    print("\n2. Training SVD model...")
    model = train_svd_model(ratings, tune_hyperparameters=False)
    
    # Evaluate
    print("\n3. Evaluating model...")
    predictions = evaluate_model(model, ratings)
    
    # Save
    print("\n4. Saving model...")
    save_model(model)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)

if __name__ == "__main__":
    main()
