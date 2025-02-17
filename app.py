from flask import Flask, request, jsonify, send_file
import cv2
import torch
import openai
import boto3
import base64
import numpy as np
import pytesseract
from PIL import Image
from io import BytesIO
import os
from ultralytics import YOLO
from waitress import serve

app = Flask(__name__)

# Load YOLOv8 model (pre-trained)
model = YOLO("yolov8s.pt")

# Secure API keys using environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")
aws_access_key = os.getenv("AWS_ACCESS_KEY")
aws_secret_key = os.getenv("AWS_SECRET_KEY")

# AWS Polly (for Text-to-Speech)
polly_client = boto3.client(
    "polly",
    aws_access_key_id=aws_access_key,
    aws_secret_access_key=aws_secret_key,
    region_name="us-east-1"
)

@app.route("/detect_objects", methods=["POST"])
def detect_objects():
    try:
        data = request.get_json()
        image_data = data.get("image", "")
        if not image_data:
            return jsonify({"error": "No image provided"}), 400

        img_bytes = base64.b64decode(image_data)
        img_array = np.array(Image.open(BytesIO(img_bytes)))

        results = model(img_array)
        detected_objects = [model.names[int(d[5])] for d in results[0].boxes.data.tolist()]

        return jsonify({"objects_detected": detected_objects})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/describe_scene", methods=["POST"])
def describe_scene():
    try:
        data = request.get_json()
        image_data = data.get("image", "")
        if not image_data:
            return jsonify({"error": "No image provided"}), 400

        response = openai.ChatCompletion.create(
            model="gpt-4-vision-preview",
            messages=[
                {"role": "system", "content": "You are an AI that describes images for visually impaired users."},
                {"role": "user", "content": "Describe the scene in this image."},
                {"role": "user", "content": {"image": image_data}}
            ]
        )

        scene_description = response["choices"][0]["message"]["content"]
        return jsonify({"scene_description": scene_description})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/speak", methods=["POST"])
def text_to_speech():
    try:
        data = request.get_json()
        text = data.get("text", "")

        if not text:
            return jsonify({"error": "No text provided"}), 400

        response = polly_client.synthesize_speech(
            Text=text,
            OutputFormat="mp3",
            VoiceId="Joanna"
        )

        audio_file = "speech_output.mp3"
        with open(audio_file, "wb") as f:
            f.write(response["AudioStream"].read())

        return send_file(audio_file, mimetype="audio/mp3", as_attachment=True)
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/read_text", methods=["POST"])
def read_text():
    try:
        data = request.get_json()
        image_data = data.get("image", "")

        if not image_data:
            return jsonify({"error": "No image provided"}), 400

        img_bytes = base64.b64decode(image_data)
        img_array = np.array(Image.open(BytesIO(img_bytes)))

        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray)

        return jsonify({"extracted_text": text})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    serve(app, host="0.0.0.0", port=port)
