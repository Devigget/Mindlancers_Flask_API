from flask import Flask, jsonify, request
from pymongo import MongoClient
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Connect to MongoDB
client = MongoClient("mongodb+srv://devigget:fz2N9FRbeRs8vyke@healthnet.yhvd5iy.mongodb.net/KickZone")
db = client["MindLancer"]

# Load SentenceTransformer model
model = SentenceTransformer("all-MiniLM-L6-v2")


def fetch_data():
    """Fetch freelancer and project data from MongoDB and preprocess."""
    freelancers = list(db["freelancers_data"].find({}, {"_id": 0}))
    projects = list(db["jobpostings"].find({}, {"_id": 0}))

    if not freelancers or not projects:
        return None, None

    freelancer_df = pd.DataFrame(freelancers)
    project_df = pd.DataFrame(projects)

    # Normalize name fields for case-insensitive matching
    freelancer_df["name"] = freelancer_df["name"].str.lower().str.strip()

    return freelancer_df, project_df


# Precompute embeddings (to avoid recomputation on every request)
freelancer_df, project_df = fetch_data()

if freelancer_df is not None and project_df is not None:
    freelancer_texts = freelancer_df.apply(lambda x:
        f"Skills: {', '.join(x['skills'])}. "
        f"Pay: {x['pay_range']}.", axis=1)
    
    project_texts = project_df.apply(lambda x:
        f"Requirements: {', '.join(x['requirements'])}. "
        f"Salary: {str(x.get('salary', 'Not specified'))}.", axis=1)

    freelancer_embeddings = model.encode(freelancer_texts.tolist(), normalize_embeddings=True)
    project_embeddings = model.encode(project_texts.tolist(), normalize_embeddings=True)

    # Compute cosine similarity matrix once
    similarity_matrix = cosine_similarity(freelancer_embeddings, project_embeddings)


def get_freelancer_recommendations(freelancer_name):
    """Get project recommendations for a specific freelancer."""
    global freelancer_df, project_df, similarity_matrix

    if freelancer_df is None or project_df is None:
        return {"error": "No freelancers or projects found"}, 500

    freelancer_name = freelancer_name.lower().strip()

    if freelancer_name not in freelancer_df["name"].values:
        return {"error": f"Freelancer '{freelancer_name}' not found"}, 404

    freelancer_idx = freelancer_df.index[freelancer_df["name"] == freelancer_name].tolist()[0]

    # Get top 3 project matches
    top_matches = np.argsort(-similarity_matrix[freelancer_idx])[:3]

    # Prepare JSON response
    recommended_projects = []
    for j in top_matches:
        recommended_projects.append({
            "title": project_df.iloc[j]['title'],
            "requirements": project_df.iloc[j]['requirements'],
            "salary": str(project_df.iloc[j]['salary'])  # Convert NumPy types to JSON-compatible format
        })

    return jsonify({
        "freelancer": freelancer_df.iloc[freelancer_idx]['name'],
        "skills": freelancer_df.iloc[freelancer_idx]['skills'],
        "top_projects": recommended_projects
    })


@app.route('/recommendations', methods=['GET'])
def recommendations():
    """API endpoint to get recommendations for a specific freelancer."""
    freelancer_name = request.args.get('freelancer_name')

    if not freelancer_name:
        return jsonify({"error": "Please provide a freelancer_name"}), 400

    return get_freelancer_recommendations(freelancer_name)


if __name__ == '__main__':
    app.run(debug=True)
