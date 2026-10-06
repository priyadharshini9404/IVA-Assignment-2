from flask import Flask, render_template, request
import cv2
import numpy as np
import base64
from deepface import DeepFace
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from PIL import Image
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

app = Flask(__name__)

mtcnn = MTCNN(image_size=160, margin=20)
facenet_model = InceptionResnetV1(pretrained="vggface2").eval()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/template-matching", methods=["GET", "POST"])
def template_matching():

    result_image = None
    score = None

    if request.method == "POST":

        main_file = request.files["main_image"]
        template_file = request.files["template_image"]

        # Read uploaded images
        main_bytes = np.frombuffer(main_file.read(), np.uint8)
        template_bytes = np.frombuffer(template_file.read(), np.uint8)

        main_image = cv2.imdecode(main_bytes, cv2.IMREAD_COLOR)
        template = cv2.imdecode(template_bytes, cv2.IMREAD_COLOR)

        # Template matching
        result = cv2.matchTemplate(
            main_image,
            template,
            cv2.TM_CCOEFF_NORMED
        )

        # Find best match
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        score = round(float(max_val), 4)

        # Get template dimensions
        template_height, template_width = template.shape[:2]

        # Top-left corner of best match
        top_left = max_loc

        # Bottom-right corner
        bottom_right = (
            top_left[0] + template_width,
            top_left[1] + template_height
        )

        # Draw rectangle
        cv2.rectangle(
            main_image,
            top_left,
            bottom_right,
            (0, 255, 0),
            3
        )

        # Convert result image to JPEG
        success, encoded_image = cv2.imencode(
            ".jpg",
            main_image
        )

        if success:
            result_image = base64.b64encode(
                encoded_image
            ).decode("utf-8")

    return render_template(
        "matching/index.html",
        result_image=result_image,
        score=score
    )

@app.route("/viola-jones", methods=["GET", "POST"])
def viola_jones():

    result_image = None
    face_count = 0

    if request.method == "POST":

        image_file = request.files["image"]

        image_bytes = np.frombuffer(
            image_file.read(),
            np.uint8
        )

        image = cv2.imdecode(
            image_bytes,
            cv2.IMREAD_COLOR
        )

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        face_count = len(faces)

        for (x, y, w, h) in faces:

            cv2.rectangle(
                image,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                3
            )

        success, encoded_image = cv2.imencode(
            ".jpg",
            image
        )

        if success:
            result_image = base64.b64encode(
                encoded_image
            ).decode("utf-8")

    return render_template(
        "viola-jones/index.html",
        result_image=result_image,
        face_count=face_count
    )
@app.route("/deepface", methods=["GET", "POST"])
def deepface():

    result_image = None
    result_text = None

    if request.method == "POST":

        image_file = request.files["image"]

        image_bytes = np.frombuffer(
            image_file.read(),
            np.uint8
        )

        image = cv2.imdecode(
            image_bytes,
            cv2.IMREAD_COLOR
        )

        result = DeepFace.analyze(
            img_path=image,
            actions=["age", "gender", "emotion"],
            enforce_detection=False
        )

        result_text = result[0]

        success, encoded_image = cv2.imencode(
            ".jpg",
            image
        )

        if success:
            result_image = base64.b64encode(
                encoded_image
            ).decode("utf-8")

    return render_template(
        "deepface/index.html",
        result_image=result_image,
        result_text=result_text
    )
@app.route("/facenet", methods=["GET", "POST"])
def facenet():

    result = None

    if request.method == "POST":

        image1_file = request.files["image1"]
        image2_file = request.files["image2"]

        image1 = Image.open(image1_file).convert("RGB")
        image2 = Image.open(image2_file).convert("RGB")

        face1 = mtcnn(image1)
        face2 = mtcnn(image2)

        if face1 is None or face2 is None:
            result = "Face not detected in one or both images."
        else:
            with torch.no_grad():
                embedding1 = facenet_model(face1.unsqueeze(0))
                embedding2 = facenet_model(face2.unsqueeze(0))

            similarity = torch.nn.functional.cosine_similarity(
                embedding1,
                embedding2
            ).item()

            if similarity > 0.6:
                result = f"Same person (Similarity: {similarity:.2f})"
            else:
                result = f"Different people (Similarity: {similarity:.2f})"

    return render_template(
        "facenet/index.html",
        result=result
    )
if __name__ == "__main__":
    app.run(debug=True)