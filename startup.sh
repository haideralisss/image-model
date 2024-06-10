#!/bin/bash

# Navigate to the application directory
source myenv/bin/activate

# Install dependencies from requirements.txt
pip install -r requirements.txt

# Run the FastAPI app with uvicorn
gunicorn --bind=0.0.0.0 main:app