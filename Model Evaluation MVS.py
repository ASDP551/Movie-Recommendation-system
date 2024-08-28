# Evaluate the hybrid model by combining both approaches
# Example: Take top-N movies from SVD predictions, then rank them using content similarity

def hybrid_recommendation(user_id, n=10):
    user_ratings = data[data['user_id'] == user_id]
    watched_movies = user_ratings['movie_id'].tolist()
    
    # Get predictions for all movies the user hasn't rated yet
    all_movie_ids = movies['movie_id'].tolist()
    not_watched = [movie for movie in all_movie_ids if movie not in watched_movies]
    
    predictions = [model.predict(user_id, movie_id) for movie_id in not_watched]
    predictions.sort(key=lambda x: x.est, reverse=True)
    
    top_n = predictions[:n]
    top_n_movie_ids = [pred.iid for pred in top_n]
    
    # Now, rank these movies based on content similarity with previously watched movies
    hybrid_scores = {}
    for movie_id in top_n_movie_ids:
        sim_score = 0
        for watched_movie in watched_movies:
            sim_score += cosine_sim[movies[movies['movie_id'] == movie_id].index[0],
                                    movies[movies['movie_id'] == watched_movie].index[0]]
        hybrid_scores[movie_id] = sim_score
    
    # Get the top-N movies based on this hybrid score
    hybrid_recommendations = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
    hybrid_movie_ids = [movie[0] for movie in hybrid_recommendations]
    return movies[movies['movie_id'].isin(hybrid_movie_ids)]

# Example: Get recommendations for user_id = 1
recommended_movies = hybrid_recommendation(user_id=1, n=10)
print(recommended_movies)
