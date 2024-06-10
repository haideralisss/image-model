from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import torch
import numpy as np
from PIL import Image   
from facenet_pytorch import InceptionResnetV1, MTCNN
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import io

app = FastAPI()

# Initialize MTCNN (face detection)
mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20)
model = InceptionResnetV1(pretrained=None)

# Storage for user embeddings
user_embeddings = {}
user_ids = []

def save_embeddings(filepath='embeddings.pkl'):
    with open(filepath, 'wb') as f:
        pickle.dump({'user_embeddings': user_embeddings, 'user_ids': user_ids}, f)
    print(f"Embeddings saved to {filepath}")

def load_model(filepath='model.pth'):
    global model
    model.load_state_dict(torch.load(filepath, map_location=torch.device('cpu')))
    model.eval()
    print(f"Model loaded from {filepath}")

def load_embeddings(filepath='embeddings.pkl'):
    global user_embeddings, user_ids
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
        user_embeddings = data['user_embeddings']
        user_ids = data['user_ids']
    print(f"Embeddings loaded from {filepath}")

def get_face_embedding(image):
    img = Image.open(io.BytesIO(image))
    img_cropped = mtcnn(img)
    
    if img_cropped is not None:
        img_embedding = model(img_cropped.unsqueeze(0))
        return img_embedding.detach().numpy()
    else:
        print(f"Face not detected in image.")
        return None

@app.post("/add_user/")
async def add_user(image: UploadFile = File(...), user_id: str = ''):
    image_data = await image.read()
    embedding = get_face_embedding(image_data)
    if embedding is not None:
        user_embeddings[user_id] = embedding
        user_ids.append(user_id)
        save_embeddings()
        return {"message": f"User {user_id} added successfully!"}
    else:
        raise HTTPException(status_code=400, detail="Face not detected in the image.")

@app.post("/predict_user/")
async def predict_user(image: UploadFile = File(...)):
    image_data = await image.read()
    load_model()
    load_embeddings()
    embedding = get_face_embedding(image_data)
    if embedding is None:
        raise HTTPException(status_code=400, detail="Face not detected in the image.")
    
    similarities = []
    for user_id in user_ids:
        stored_embedding = user_embeddings[user_id]
        sim = cosine_similarity(embedding, stored_embedding)
        similarities.append((user_id, sim[0][0]))
    
    if similarities:
        best_match = max(similarities, key=lambda x: x[1])
        return {"user_id": best_match[0]}
    else:
        raise HTTPException(status_code=404, detail="No users available for matching.")
