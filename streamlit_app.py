import streamlit as st
import cv2
import numpy as np
from deepface import DeepFace
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from PIL import Image


# ---------------- PAGE SETTINGS ----------------

st.set_page_config(
    page_title="IVA VisionLab",
    page_icon="🔬",
    layout="wide"
)
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #eef4ff, #f8fbff);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #12355b, #1d5d8f);
    }

    [data-testid="stSidebar"] * {
        color: white;
    }

    h1 {
        color: #12355b;
        font-weight: 700;
    }

    h2, h3 {
        color: #1d5d8f;
    }

    .stButton > button {
        background-color: #1d5d8f;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
        font-weight: 600;
    }

    .stButton > button:hover {
        background-color: #12355b;
        color: white;
    }
</style>
""")

st.title("🔬 IVA VisionLab")
st.subheader("Image & Video Analytics Laboratory")
st.write("Explore classical and deep-learning based image analysis techniques.")

st.divider()


# ---------------- LOAD FACE MODELS ----------------

@st.cache_resource
def load_models():

    mtcnn = MTCNN(image_size=160, margin=20)

    facenet_model = InceptionResnetV1(
        pretrained="vggface2"
    ).eval()

    return mtcnn, facenet_model


mtcnn, facenet_model = load_models()


# ---------------- SIDEBAR ----------------

st.sidebar.title("IVA Operators")

operator = st.sidebar.selectbox(
    "Select an operator",
    [
        "Template Matching",
        "Viola–Jones",
        "DeepFace",
        "FaceNet"
    ]
)


# =================================================
# TEMPLATE MATCHING
# =================================================

if operator == "Template Matching":

    st.header("Template Matching")

    main_file = st.file_uploader(
        "Upload Main Image",
        type=["jpg", "jpeg", "png"],
        key="main"
    )

    template_file = st.file_uploader(
        "Upload Template Image",
        type=["jpg", "jpeg", "png"],
        key="template"
    )

    if main_file and template_file:

        main_bytes = np.frombuffer(
            main_file.read(),
            np.uint8
        )

        template_bytes = np.frombuffer(
            template_file.read(),
            np.uint8
        )

        main_image = cv2.imdecode(
            main_bytes,
            cv2.IMREAD_COLOR
        )

        template = cv2.imdecode(
            template_bytes,
            cv2.IMREAD_COLOR
        )

        result = cv2.matchTemplate(
            main_image,
            template,
            cv2.TM_CCOEFF_NORMED
        )

        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        h, w = template.shape[:2]

        cv2.rectangle(
            main_image,
            max_loc,
            (max_loc[0] + w, max_loc[1] + h),
            (0, 255, 0),
            3
        )

        st.image(
            cv2.cvtColor(main_image, cv2.COLOR_BGR2RGB),
            caption="Matched Template"
        )

        st.success(
            f"Matching Score: {max_val:.4f}"
        )


# =================================================
# VIOLA–JONES
# =================================================

elif operator == "Viola–Jones":

    st.header("Viola–Jones Face Detection")

    image_file = st.file_uploader(
        "Upload an Image",
        type=["jpg", "jpeg", "png"],
        key="viola"
    )

    if image_file:

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

        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades
            + "haarcascade_frontalface_default.xml"
        )

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        for (x, y, w, h) in faces:

            cv2.rectangle(
                image,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                3
            )

        st.image(
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            caption="Detected Faces"
        )

        st.success(
            f"Faces Detected: {len(faces)}"
        )


# =================================================
# DEEPFACE
# =================================================

elif operator == "DeepFace":

    st.header("DeepFace Analysis")

    image_file = st.file_uploader(
        "Upload an Image",
        type=["jpg", "jpeg", "png"],
        key="deepface"
    )

    if image_file:

        image_bytes = np.frombuffer(
            image_file.read(),
            np.uint8
        )

        image = cv2.imdecode(
            image_bytes,
            cv2.IMREAD_COLOR
        )

        if st.button("Analyze Face"):

            with st.spinner("Analyzing..."):

                result = DeepFace.analyze(
                    img_path=image,
                    actions=[
                        "age",
                        "gender",
                        "emotion"
                    ],
                    enforce_detection=False
                )

            result = result[0]

            st.image(
                cv2.cvtColor(
                    image,
                    cv2.COLOR_BGR2RGB
                )
            )

            st.subheader("Analysis Result")

            st.write(
                f"Age: {result['age']}"
            )

            st.write(
                f"Gender: {result['dominant_gender']}"
            )

            st.write(
                f"Emotion: {result['dominant_emotion']}"
            )


# =================================================
# FACENET
# =================================================

elif operator == "FaceNet":

    st.header("FaceNet Face Verification")

    image1_file = st.file_uploader(
        "Upload First Face",
        type=["jpg", "jpeg", "png"],
        key="face1"
    )

    image2_file = st.file_uploader(
        "Upload Second Face",
        type=["jpg", "jpeg", "png"],
        key="face2"
    )

    if image1_file and image2_file:

        if st.button("Compare Faces"):

            image1 = Image.open(
                image1_file
            ).convert("RGB")

            image2 = Image.open(
                image2_file
            ).convert("RGB")

            face1 = mtcnn(image1)
            face2 = mtcnn(image2)

            if face1 is None or face2 is None:

                st.error(
                    "Face not detected in one or both images."
                )

            else:

                with torch.no_grad():

                    embedding1 = facenet_model(
                        face1.unsqueeze(0)
                    )

                    embedding2 = facenet_model(
                        face2.unsqueeze(0)
                    )

                similarity = torch.nn.functional.cosine_similarity(
                    embedding1,
                    embedding2
                ).item()

                st.metric(
                    "Similarity Score",
                    f"{similarity:.2f}"
                )

                if similarity > 0.6:

                    st.success(
                        "Same person"
                    )

                else:

                    st.error(
                        "Different people"
                    )