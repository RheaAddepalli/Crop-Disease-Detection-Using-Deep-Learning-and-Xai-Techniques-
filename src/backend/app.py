from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import cv2
import sys
import time

# ✅ app.py is in src/backend/ → go up 3 levels to reach project root
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ✅ frontend is in src/frontend/ (sibling of backend/)
FRONTEND_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend"))

# ✅ predict_xai.py is in src/
sys.path.append(os.path.join(ROOT_DIR, "src"))

from predict_xai import predict_for_web

# ✅ Flask serves static files from frontend/
app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")
CORS(app)

# ✅ temp folder stays inside src/backend/
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ✅ sample images are in datasets/sample_web_images/
SAMPLE_DIR = os.path.join(ROOT_DIR, "datasets", "sample_web_images")


# ✅ Serve index.html at root URL
@app.route("/")
def home():
    return send_from_directory(FRONTEND_DIR, "index.html")


# ✅ Serve sample images
@app.route("/samples/<filename>")
def get_sample(filename):
    return send_from_directory(SAMPLE_DIR, filename)


# ✅ List all sample images
@app.route("/samples")
def list_samples():
    files = sorted([
        f for f in os.listdir(SAMPLE_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    return jsonify({"samples": files})


@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    result = predict_for_web(path)

    predicted_class = result["label"]
    confidence = result["confidence"]
    grad_img = result["image"]

    timestamp = int(time.time())
    grad_filename = f"gradcam_{timestamp}.jpg"
    grad_path = os.path.join(UPLOAD_FOLDER, grad_filename)
    cv2.imwrite(grad_path, grad_img)

    return jsonify({
        "prediction": predicted_class,
        "confidence": confidence,
        "gradcam": f"/temp/{grad_filename}"
    })

@app.route("/temp/<filename>")
def get_image(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


if __name__ == "__main__":
    app.run(debug=True)























# without confidence
# from flask import Flask, request, jsonify, send_from_directory
# from flask_cors import CORS
# import os
# import cv2
# import sys
# import time

# # ✅ app.py is in src/backend/ → go up 3 levels to reach project root
# ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# # ✅ frontend is in src/frontend/ (sibling of backend/)
# FRONTEND_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend"))

# # ✅ predict_xai.py is in src/
# sys.path.append(os.path.join(ROOT_DIR, "src"))

# from predict_xai import predict_for_web

# # ✅ Flask serves static files from frontend/
# app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")
# CORS(app)

# # ✅ temp folder stays inside src/backend/
# UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp")
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# # ✅ sample images are in datasets/sample_web_images/
# SAMPLE_DIR = os.path.join(ROOT_DIR, "datasets", "sample_web_images")


# # ✅ Serve index.html at root URL
# @app.route("/")
# def home():
#     return send_from_directory(FRONTEND_DIR, "index.html")


# # ✅ Serve sample images
# @app.route("/samples/<filename>")
# def get_sample(filename):
#     return send_from_directory(SAMPLE_DIR, filename)


# # ✅ List all sample images
# @app.route("/samples")
# def list_samples():
#     files = sorted([
#         f for f in os.listdir(SAMPLE_DIR)
#         if f.lower().endswith((".jpg", ".jpeg", ".png"))
#     ])
#     return jsonify({"samples": files})


# @app.route("/predict", methods=["POST"])
# def predict():

#     if "image" not in request.files:
#         return jsonify({"error": "No file uploaded"}), 400

#     file = request.files["image"]

#     if file.filename == "":
#         return jsonify({"error": "Empty filename"}), 400

#     path = os.path.join(UPLOAD_FOLDER, file.filename)
#     file.save(path)

#     predicted_class, grad_img = predict_for_web(path)

#     # Timestamp makes filename unique → fixes browser cache problem
#     timestamp = int(time.time())
#     grad_filename = f"gradcam_{timestamp}.jpg"
#     grad_path = os.path.join(UPLOAD_FOLDER, grad_filename)
#     cv2.imwrite(grad_path, grad_img)

#     return jsonify({
#         "prediction": predicted_class,
        
#         "gradcam": f"/temp/{grad_filename}"
#     })


# @app.route("/temp/<filename>")
# def get_image(filename):
#     return send_from_directory(UPLOAD_FOLDER, filename)


# if __name__ == "__main__":
#     app.run(debug=True)