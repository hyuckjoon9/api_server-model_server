from flask import Flask, request, jsonify, send_file
import numpy as np
import cv2
import io
import tensorflow as tf

app = Flask(__name__)

# 여기는 만든 모델 - 실제 모델 경로 넣어야 함
# model = tf.keras.models.load_model(".h5")


# URL - post
@app.route("/predict", methods=["POST"])
def predict():
    # 이미지 받음
    if "file" not in request.files:
        return jsonify({"error": "파일이 제공되지 않음"}), 400

    file = request.files["file"]
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "이미지 디코딩 실패"}), 400

    _, img_encoded = cv2.imencode(".jpg", img)
    byte_io = io.BytesIO(img_encoded.tobytes())

    return send_file(byte_io, mimetype="image/jpeg")


if __name__ == "__main__":
    app.run(debug=True)
