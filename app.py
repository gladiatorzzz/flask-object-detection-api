from flask import Flask, request, jsonify
import cv2
import torch
import openai
import boto3
import base64
import numpy as np
from PIL import Image
from io import BytesIO

app = Flask(__name__)

# Load YOLOv8 model (pre-trained)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# OpenAI API Key (for Scene Description)
openai.api_key = "your_openai_api_key"

# AWS Polly (for Text-to-Speech)
polly_client = boto3.client(
    "polly",
    aws_access_key_id="your_aws_access_key",
    aws_secret_access_key="your_aws_secret_key",
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

        detected_objects = []
        for obj in results.pandas().xyxy[0].itertuples():
            detected_objects.append(obj.name)

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
                {"role": "user", "content": image_data}
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

        return jsonify({"audio_file": "speech_output.mp3"})

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


if __name__ == '__main__':
    import os
    from waitress import serve

    port = int(os.environ.get("PORT", 5000))  
    serve(app, host="0.0.0.0", port=port)




