from flask import Flask, request, jsonify
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import InceptionResnetV1, MTCNN
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import io
import os

app = Flask(__name__)

# Initialize MTCNN (face detection)
mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20)
model = InceptionResnetV1(pretrained=None)

# Storage for user embeddings
user_embeddings = {}
user_ids = []

def save_embeddings(filepath='embeddings.pkl'):
    try:
        with open(filepath, 'wb') as f:
            pickle.dump({'user_embeddings': user_embeddings, 'user_ids': user_ids}, f)
        print(f"Embeddings saved to {filepath}")
    except Exception as e:
        print(f"Error saving embeddings: {e}")

def load_model(filepath='model.pth'):
    global model
    if os.path.exists(filepath):
        try:
            model.load_state_dict(torch.load(filepath, map_location=torch.device('cpu')))
            model.eval()
            print(f"Model loaded from {filepath}")
        except Exception as e:
            print(f"Error loading model: {e}")
    else:
        print(f"Model file {filepath} does not exist.")

def load_embeddings(filepath='embeddings.pkl'):
    global user_embeddings, user_ids
    if os.path.exists(filepath):
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                user_embeddings = data['user_embeddings']
                user_ids = data['user_ids']
            print(f"Embeddings loaded from {filepath}")
        except Exception as e:
            print(f"Error loading embeddings: {e}")
    else:
        print(f"Embeddings file {filepath} does not exist.")

def get_face_embedding(image):
    try:
        img = Image.open(io.BytesIO(image))
        img_cropped = mtcnn(img)
        
        if img_cropped is not None:
            img_embedding = model(img_cropped.unsqueeze(0))
            return img_embedding.detach().numpy()
        else:
            print("Face not detected in image.")
            return None
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

@app.route("/add_user/", methods=['POST'])
def add_user():
    if 'image' not in request.files or 'user_id' not in request.form:
        return jsonify({"error": "Image file and user_id are required"}), 400
    
    image_file = request.files['image']
    user_id = request.form['user_id']
    image_data = image_file.read()
    embedding = get_face_embedding(image_data)
    
    if embedding is not None:
        user_embeddings[user_id] = embedding
        user_ids.append(user_id)
        save_embeddings()
        return jsonify({"message": f"User {user_id} added successfully!"})
    else:
        return jsonify({"error": "Face not detected in the image"}), 400

@app.route("/predict_user/", methods=['POST'])
def predict_user():
    if 'image' not in request.files:
        return jsonify({"error": "Image file is required"}), 400
    
    image_file = request.files['image']
    image_data = image_file.read()
    load_model()
    load_embeddings()
    embedding = get_face_embedding(image_data)
    
    if embedding is None:
        return jsonify({"error": "Face not detected in the image"}), 400
    
    similarities = []
    for user_id in user_ids:
        stored_embedding = user_embeddings[user_id]
        if stored_embedding.shape == embedding.shape:
            sim = cosine_similarity(embedding, stored_embedding)
            similarities.append((user_id, sim[0][0]))
        else:
            print(f"Shape mismatch: {embedding.shape} vs {stored_embedding.shape}")
    
    if similarities:
        best_match = max(similarities, key=lambda x: x[1])
        return jsonify({"user_id": best_match[0]})
    else:
        return jsonify({"error": "No users available for matching"}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
