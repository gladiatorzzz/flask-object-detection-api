import os
from flask import Flask

app = Flask(__name__)

@app.route("/")
def home():
    return "Flask Object Detection API is Running!"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Get PORT from environment, default to 5000
    app.run(host="0.0.0.0", port=port)  # Bind to 0.0.0.0 for external access

