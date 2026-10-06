import streamlit as st
import cv2
import numpy as np
from deepface import DeepFace
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from PIL import Image

# ---------------------------------------------------------
# PAGE SETTINGS
# ---------------------------------------------------------
st.set_page_config(
    page_title="IVA VisionLab",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# CUSTOM STYLING
# ---------------------------------------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #eef4ff 0%, #f8fbff 100%);
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #12355b 0%, #1d5d8f 100%);
}

[data-testid="stSidebar"] * {
    color: white;
}

[data-testid="stSidebar"] .stSelectbox label {
    color: white !important;
    font-weight: 600;
}

.hero {
    background: linear-gradient(135deg, #12355b, #2878b5);
    padding: 28px 32px;
    border-radius: 18px;
    color: white;
    margin-bottom: 22px;
    box-shadow: 0 8px 25px rgba(18, 53, 91, 0.18);
}

.hero h1 {
    color: white !important;
    margin-bottom: 5px;
    font-size: 38px;
}

.hero p {
    color: #eaf4ff;
    font-size: 17px;
    margin-bottom: 0;
}

.feature-card {
    background: white;
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #dbe8f5;
    box-shadow: 0 5px 18px rgba(31, 70, 110, 0.08);
    min-height: 150px;
}

.feature-card h3 {
    color: #12355b;
    margin-top: 0;
}

.feature-card p {
    color: #526579;
    line-height: 1.55;
}

.section-card {
    background: white;
    padding: 24px 28px;
    border-radius: 16px;
    border: 1px solid #dbe8f5;
    box-shadow: 0 5px 18px rgba(31, 70, 110, 0.07);
    margin-bottom: 20px;
}

.section-card h2 {
    color: #12355b;
    margin-top: 0;
}

.tip {
    background: #eaf4ff;
    border-left: 5px solid #2878b5;
    padding: 12px 16px;
    border-radius: 8px;
    color: #23415c;
    margin: 15px 0;
}

.stButton > button {
    background-color: #1d5d8f;
    color: white;
    border-radius: 9px;
    border: none;
    padding: 10px 22px;
    font-weight: 600;
}

.stButton > button:hover {
    background-color: #12355b;
    color: white;
}

h1, h2, h3 {
    color: #12355b;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------
st.markdown("""
<div class="hero">
    <h1>🔬 IVA VisionLab</h1>
    <p>Image & Video Analytics Laboratory</p>
    <p>Explore classical computer vision and deep-learning based face analysis techniques.</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
st.sidebar.title("IVA Operators")
st.sidebar.caption("Choose a computer vision technique")

operator = st.sidebar.selectbox(
    "Select an operator",
    [
        "Template Matching",
        "Viola–Jones",
        "DeepFace",
        "FaceNet"
    ]
)

# ---------------------------------------------------------
# LOAD FACENET MODELS
# ---------------------------------------------------------
@st.cache_resource
def load_models():
    mtcnn = MTCNN(image_size=160, margin=20)
    facenet_model = InceptionResnetV1(
        pretrained="vggface2"
    ).eval()
    return mtcnn, facenet_model

mtcnn, facenet_model = load_models()

# ---------------------------------------------------------
# TEMPLATE MATCHING
# ---------------------------------------------------------
if operator == "Template Matching":

    st.markdown("""
    <div class="section-card">
        <h2>🔎 Template Matching</h2>
        <p>
        Template Matching is a classical image-processing technique used to
        locate a smaller image or pattern inside a larger image.
        The template is compared with different regions of the main image
        to find the best matching location.
        </p>
        <div class="tip">
        <b>How it works:</b> Upload a main image and a smaller template.
        The system finds the location with the highest matching score
        and highlights it with a rectangle.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        main_file = st.file_uploader(
            "📷 Upload Main Image",
            type=["jpg", "jpeg", "png"],
            key="main"
        )

    with col2:
        template_file = st.file_uploader(
            "🧩 Upload Template Image",
            type=["jpg", "jpeg", "png"],
            key="template"
        )

    if main_file and template_file:

        main_bytes = np.frombuffer(main_file.read(), np.uint8)
        template_bytes = np.frombuffer(template_file.read(), np.uint8)

        main_image = cv2.imdecode(main_bytes, cv2.IMREAD_COLOR)
        template = cv2.imdecode(template_bytes, cv2.IMREAD_COLOR)

        if (
            main_image is None
            or template is None
            or template.shape[0] > main_image.shape[0]
            or template.shape[1] > main_image.shape[1]
        ):
            st.error("Please make sure the template is smaller than the main image.")
        else:
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
                caption="Best Matching Region",
                use_container_width=True
            )

            st.success(f"Matching Score: {max_val:.4f}")

# ---------------------------------------------------------
# VIOLA-JONES
# ---------------------------------------------------------
elif operator == "Viola–Jones":

    st.markdown("""
    <div class="section-card">
        <h2>👤 Viola–Jones Face Detection</h2>
        <p>
        Viola–Jones is a classical real-time object detection algorithm.
        It uses Haar-like features, an integral image, AdaBoost and a
        cascade classifier to quickly detect objects such as faces.
        </p>
        <div class="tip">
        <b>How it works:</b> Upload a face image. The Haar Cascade
        classifier scans the image and draws a rectangle around each
        detected face.
        </div>
    </div>
    """, unsafe_allow_html=True)

    image_file = st.file_uploader(
        "📷 Upload an Image",
        type=["jpg", "jpeg", "png"],
        key="viola"
    )

    if image_file:

        image_bytes = np.frombuffer(image_file.read(), np.uint8)
        image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

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
            caption="Detected Faces",
            use_container_width=True
        )

        st.success(f"Faces Detected: {len(faces)}")

# ---------------------------------------------------------
# DEEPFACE
# ---------------------------------------------------------
elif operator == "DeepFace":

    st.markdown("""
    <div class="section-card">
        <h2>🧠 DeepFace</h2>
        <p>
        DeepFace is a deep-learning based face analysis framework.
        It can analyze a face image and estimate attributes such as
        age, gender and dominant emotion.
        </p>
        <div class="tip">
        <b>How it works:</b> Upload a clear face image and click
        <b>Analyze Face</b>. The trained deep-learning models process
        the face and display the analysis results.
        </div>
    </div>
    """, unsafe_allow_html=True)

    image_file = st.file_uploader(
        "📷 Upload a Face Image",
        type=["jpg", "jpeg", "png"],
        key="deepface"
    )

    if image_file:

        image_bytes = np.frombuffer(image_file.read(), np.uint8)
        image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

        if st.button("🧠 Analyze Face"):

            with st.spinner("Analyzing the face..."):
                result = DeepFace.analyze(
                    img_path=image,
                    actions=["age", "gender", "emotion"],
                    enforce_detection=False
                )

            if isinstance(result, list):
                result = result[0]

            st.image(
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                caption="Input Face",
                use_container_width=True
            )

            st.subheader("Analysis Result")

            c1, c2, c3 = st.columns(3)

            with c1:
                st.metric("Estimated Age", result["age"])

            with c2:
                st.metric("Gender", result["dominant_gender"])

            with c3:
                st.metric("Emotion", result["dominant_emotion"])

# ---------------------------------------------------------
# FACENET
# ---------------------------------------------------------
elif operator == "FaceNet":

    st.markdown("""
    <div class="section-card">
        <h2>🧬 FaceNet Face Verification</h2>
        <p>
        FaceNet is a deep-learning based face recognition method.
        It converts a face into a numerical representation called
        an embedding. Two embeddings can then be compared to measure
        how similar the faces are.
        </p>
        <div class="tip">
        <b>How it works:</b> Upload two face images. FaceNet creates
        embeddings for both images and calculates their cosine similarity.
        A higher similarity indicates that the faces are more likely
        to belong to the same person.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        image1_file = st.file_uploader(
            "👤 Upload First Face",
            type=["jpg", "jpeg", "png"],
            key="face1"
        )

    with col2:
        image2_file = st.file_uploader(
            "👤 Upload Second Face",
            type=["jpg", "jpeg", "png"],
            key="face2"
        )

    if image1_file and image2_file:

        if st.button("🔍 Compare Faces"):

            image1 = Image.open(image1_file).convert("RGB")
            image2 = Image.open(image2_file).convert("RGB")

            face1 = mtcnn(image1)
            face2 = mtcnn(image2)

            if face1 is None or face2 is None:
                st.error("Face not detected in one or both images.")

            else:
                with torch.no_grad():
                    embedding1 = facenet_model(face1.unsqueeze(0))
                    embedding2 = facenet_model(face2.unsqueeze(0))

                similarity = torch.nn.functional.cosine_similarity(
                    embedding1,
                    embedding2
                ).item()

                st.metric(
                    "Cosine Similarity Score",
                    f"{similarity:.2f}"
                )

                if similarity > 0.6:
                    st.success("✅ Same person")
                else:
                    st.error("❌ Different people")
