"""
Hybrid Recommendation Model + Evaluation Metrics
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

class HybridRecommender:
    """
    Hybrid recommender combining collaborative and content-based filtering
    """
    
    def __init__(self, svd_model, content_model, item_rating_counts, cold_start_threshold=5):
        """
        Initialize hybrid recommender focusing on Item Cold-Start
        """
        self.svd_model = svd_model
        self.tfidf = content_model['tfidf']
        self.tfidf_matrix = content_model['tfidf_matrix']
        self.item_ids = content_model['item_ids']
        self.item_rating_counts = item_rating_counts
        self.cold_start_threshold = cold_start_threshold
        
    def predict_hybrid(self, user_id, item_id, user_profile):
        """
        Predict rating using item-focused hybrid logic
        """
        # Collaborative filtering prediction
        try:
            collab_pred = self.svd_model.predict(user_id, item_id).est
        except:
            collab_pred = 3.0
            
        # Get item rating count
        item_ratings_count = self.item_rating_counts.get(item_id, 0)
        
        # Calculate content score
        try:
            idx = np.where(self.item_ids == item_id)[0][0]
            item_vector = self.tfidf_matrix[idx].toarray()[0]
            sim = np.dot(user_profile, item_vector) / (np.linalg.norm(user_profile) * np.linalg.norm(item_vector) + 1e-8)
            content_score = 1 + (sim * 4)  # map to 1-5 scale
        except:
            content_score = 3.0
        
        if item_ratings_count < self.cold_start_threshold:
            # Cold-start item: Heavy content-based
            return 0.2 * collab_pred + 0.8 * content_score
        else:
            # Warm item: Heavy collaborative
            return 0.8 * collab_pred + 0.2 * content_score
    
    def recommend(self, user_id, user_ratings, n=10):
        rated_items = set([item_id for item_id, _ in user_ratings])
        all_items = self.item_ids
        unrated_items = [item for item in all_items if item not in rated_items]
        
        # Build user profile for content similarity
        liked_items = [i for i, r in user_ratings if r >= 4]
        if not liked_items:
            liked_items = [i for i, r in user_ratings]
            
        user_profile = np.zeros(self.tfidf_matrix.shape[1])
        for i in liked_items:
            idx = np.where(self.item_ids == i)[0]
            if len(idx) > 0:
                user_profile += self.tfidf_matrix[idx[0]].toarray()[0]
        
        if len(liked_items) > 0:
            user_profile /= len(liked_items)
            
        predictions = []
        for item_id in unrated_items:
            is_cold_start = self.item_rating_counts.get(item_id, 0) < self.cold_start_threshold
            pred_rating = self.predict_hybrid(user_id, item_id, user_profile)
            predictions.append((item_id, pred_rating, is_cold_start))
        
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n]


def precision_at_k(recommended, relevant, k=10):
    """
    Precision@K: Of top K recommendations, how many are relevant?
    
    Args:
        recommended: List of recommended item IDs (ordered)
        relevant: Set of relevant item IDs
        k: Cutoff position
    
    Returns:
        Precision@K score (0-1)
    """
    if k == 0:
        return 0.0
    
    recommended_at_k = recommended[:k]
    relevant_recommended = len([item for item in recommended_at_k if item in relevant])
    
    return relevant_recommended / k


def recall_at_k(recommended, relevant, k=10):
    """
    Recall@K: Of all relevant items, how many in top K?
    
    Args:
        recommended: List of recommended item IDs (ordered)
        relevant: Set of relevant item IDs
        k: Cutoff position
    
    Returns:
        Recall@K score (0-1)
    """
    if len(relevant) == 0:
        return 0.0
    
    recommended_at_k = recommended[:k]
    relevant_recommended = len([item for item in recommended_at_k if item in relevant])
    
    return relevant_recommended / len(relevant)


def ndcg_at_k(recommended, relevant, k=10):
    """
    Normalized Discounted Cumulative Gain@K
    Rewards relevant items appearing higher in the ranking
    
    Args:
        recommended: List of recommended item IDs (ordered)
        relevant: Set of relevant item IDs
        k: Cutoff position
    
    Returns:
        NDCG@K score (0-1)
    """
    recommended_at_k = recommended[:k]
    
    # DCG: sum of (relevance / log2(position + 1))
    dcg = 0.0
    for i, item in enumerate(recommended_at_k):
        if item in relevant:
            dcg += 1.0 / np.log2(i + 2)  # i+2 because positions start at 1
    
    # IDCG: best possible DCG (all relevant items at top)
    idcg = 0.0
    for i in range(min(len(relevant), k)):
        idcg += 1.0 / np.log2(i + 2)
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def evaluate_recommender(model, test_ratings, k=10, relevance_threshold=4):
    """
    Evaluate recommender system on test set
    
    Args:
        model: HybridRecommender instance
        test_ratings: DataFrame with user_id, item_id, rating
        k: Cutoff for metrics
        relevance_threshold: Min rating to consider item relevant
    
    Returns:
        Dict with evaluation metrics
    """
    print(f"\nEvaluating with k={k}, relevance_threshold={relevance_threshold}")
    
    # Group ratings by user
    user_ratings = test_ratings.groupby('user_id').apply(
        lambda x: list(zip(x['item_id'], x['rating']))
    ).to_dict()
    
    all_precision = []
    all_recall = []
    all_ndcg = []
    cold_start_users = 0
    
    for user_id, ratings in user_ratings.items():
        if len(ratings) < 2:  # Need at least 2 ratings (1 to train, 1 to test)
            continue
        
        # Split into "known" and "test" ratings
        train_ratings = ratings[:-3] if len(ratings) > 3 else ratings[:-1]
        test_ratings_user = ratings[-3:] if len(ratings) > 3 else ratings[-1:]
        
        # Get recommendations
        try:
            recommendations = model.recommend(user_id, train_ratings, n=k)
            recommended_items = [item_id for item_id, _, _ in recommendations]
            
            # Relevant items (high ratings in test set)
            relevant_items = set([
                item_id for item_id, rating in test_ratings_user 
                if rating >= relevance_threshold
            ])
            
            if len(relevant_items) == 0:
                continue
            
            # Calculate metrics
            p_at_k = precision_at_k(recommended_items, relevant_items, k)
            r_at_k = recall_at_k(recommended_items, relevant_items, k)
            ndcg = ndcg_at_k(recommended_items, relevant_items, k)
            
            all_precision.append(p_at_k)
            all_recall.append(r_at_k)
            all_ndcg.append(ndcg)
            
            if len(train_ratings) < model.cold_start_threshold:
                cold_start_users += 1
                
        except Exception as e:
            continue
    
    results = {
        'precision@k': np.mean(all_precision) if all_precision else 0,
        'recall@k': np.mean(all_recall) if all_recall else 0,
        'ndcg@k': np.mean(all_ndcg) if all_ndcg else 0,
        'num_users_evaluated': len(all_precision),
        'cold_start_users': cold_start_users
    }
    
    return results


def print_evaluation_results(results, model_name="Model"):
    """Pretty print evaluation results"""
    print(f"\n{'=' * 60}")
    print(f"{model_name} Performance @ K={10}")
    print(f"{'=' * 60}")
    print(f"Precision@10:  {results['precision@k']:.4f}")
    print(f"Recall@10:     {results['recall@k']:.4f}")
    print(f"NDCG@10:       {results['ndcg@k']:.4f}")
    print(f"\nUsers evaluated: {results['num_users_evaluated']}")
    print(f"Cold-start users: {results['cold_start_users']}")
    print(f"{'=' * 60}")


def main():
    """Evaluate hybrid model"""
    print("=" * 60)
    print("HYBRID MODEL EVALUATION")
    print("=" * 60)
    
    # Load models
    print("\n1. Loading models...")
    with open('models/svd_model.pkl', 'rb') as f:
        svd_model = pickle.load(f)
    
    with open('models/content_model.pkl', 'rb') as f:
        content_model = pickle.load(f)
    
    print("   ✓ Models loaded")
    
    # Load test data
    print("\n2. Loading test data...")
    ratings = pd.read_csv(
        'data/ml-100k/u.data',
        sep='\t',
        names=['user_id', 'item_id', 'rating', 'timestamp']
    )
    
    # Time-based split (last 20% of data)
    ratings = ratings.sort_values('timestamp')
    split_idx = int(len(ratings) * 0.8)
    test_ratings = ratings[split_idx:]
    
    print(f"   Test set: {len(test_ratings):,} ratings")
    
    # Compute item rating counts
    item_rating_counts = ratings.groupby('item_id').size().to_dict()
    
    # Create hybrid model
    print("\n3. Creating hybrid recommender...")
    hybrid = HybridRecommender(svd_model, content_model, item_rating_counts, cold_start_threshold=5)
    
    # Evaluate
    print("\n4. Evaluating...")
    results = evaluate_recommender(hybrid, test_ratings, k=10)
    
    # Print results
    print_evaluation_results(results, "Hybrid Recommender")
    
    # Save hybrid model
    print("\n5. Saving hybrid model...")
    model_data = {
        'svd_model': svd_model,
        'content_model': content_model,
        'item_rating_counts': item_rating_counts,
        'cold_start_threshold': 5
    }
    
    with open('models/hybrid_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    print("   ✓ Hybrid model saved to models/hybrid_model.pkl")
    
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
