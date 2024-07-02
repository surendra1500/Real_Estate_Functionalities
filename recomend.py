from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')  # Replace with your MongoDB connection string
db = client['real_estate_db']  # Replace with your database name
collection = db['listings']  # Replace with your collection name

def fetch_listings(moduleId, categoryId, subCategoryId, countryId, city, macroCity, zip_code):
    # Fetch listings that match the fixed parameters
    query = {
        'moduleId': {'$in': moduleId},
        'categoryId': categoryId,
        'subCategoryId': {'$in': subCategoryId},
        'countryId': countryId,
        'city': city,
        'macroCity': macroCity,
        'zip': zip_code
    }
    listings = list(collection.find(query))
    return listings

def combine_features(row, weights):
    # Handle missing values and combine features with weights
    budgets = " ".join(row['budgets']) if row['budgets'] else ""
    priceTypeId = row['priceTypeId'] if row['priceTypeId'] else ""
    attributes = " ".join(row['attributes']) if row['attributes'] else ""
    latitude = str(row['latitude']) if row['latitude'] else ""
    longitude = str(row['longitude']) if row['longitude'] else ""
    
    combined_features = f"{weights['budgets'] * budgets} {weights['priceTypeId'] * priceTypeId} {weights['attributes'] * attributes} {weights['latitude'] * latitude} {weights['longitude'] * longitude}"
    return combined_features

def compute_similarity(listings, input_params, weights):
    # Create a DataFrame from the listings
    df = pd.DataFrame(listings)
    
    # Combine the relevant fields into a single string for each listing with weights
    df['combined_features'] = df.apply(lambda x: combine_features(x, weights), axis=1)
    
    # Combine input parameters into a single string with weights
    input_combined_features = combine_features(input_params, weights)
    
    # Vectorize the features
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df['combined_features'])
    
    # Vectorize the input features
    input_tfidf = vectorizer.transform([input_combined_features])
    
    # Compute cosine similarity
    cosine_sim = cosine_similarity(input_tfidf, tfidf_matrix)
    
    # Get the indices of the most similar listings
    similar_indices = cosine_sim[0].argsort()[-5:][::-1]  # Top 5 most similar listings
    
    similar_listings = df.iloc[similar_indices]
    return similar_listings

def recommend_listings(input_params, weights):
    # Fetch listings that match the fixed parameters
    listings = fetch_listings(
        moduleId=input_params['moduleId'],
        categoryId=input_params['categoryId'],
        subCategoryId=input_params['subCategoryId'],
        countryId=input_params['countryId'],
        city=input_params['city'],
        macroCity=input_params['macroCity'],
        zip_code=input_params['zip']
    )
    
    if not listings:
        return "No listings found with the given parameters."
    
    # Compute similarity and get the most similar listings
    similar_listings_df = compute_similarity(listings, input_params, weights)
    
    # Get the full listings details for the most similar listings
    similar_listings = []
    for index, row in similar_listings_df.iterrows():
        similar_listings.append(listings[index])
    
    return similar_listings

# Example input parameters
input_params = {
    "moduleId": ["Qd3pAp"],
    "categoryId": "YoRiob",
    "subCategoryId": ["XODss1", "BvF8qq"],
    "budgets": [],
    "priceTypeId": "hCDjcS",
    "attributes": [],
    "countryId": "hCDjcS",
    "latitude": "17.4096384",
    "longitude": "78.4728064",
    "city": "Hyderabad",
    "macroCity": "Hyderabad",
    "zip": "500022"
}

# Weights for the features
weights = {
    'budgets': 1.0,
    'priceTypeId': 1.0,
    'attributes': 1.0,
    'latitude': 2.0,  # More importance to proximity
    'longitude': 2.0  # More importance to proximity
}

# Get recommendations
recommendations = recommend_listings(input_params, weights)
for recommendation in recommendations:
    print(recommendation)
