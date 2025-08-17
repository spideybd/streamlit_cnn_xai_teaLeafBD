import streamlit as st
import torch
from PIL import Image
import numpy as np
import os
import io
import zipfile
from torchvision import transforms

from models import load_model
from xai_utils import generate_cam_visualizations, generate_lime_visualization

st.set_page_config(
    page_title="Tea Leaf XAI Explainer",
    layout="wide"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = [
    "Tea algal leaf spot", "Gray Blight", "Brown Blight", "Green mirid bug",
    "Red spider", "Helopeltis", "Healthy leaf"
]
NUM_CLASSES = len(CLASS_NAMES)
WEIGHTS_DIR = "weights"
SAMPLES_DIR = "sample_images"

MODELS_INFO = {
    "CustomCNN": {
        "path": os.path.join(WEIGHTS_DIR, "custom_cnn.pth"),
        "architecture": "Custom CNN (VGG-style)",
        "input_size": "224x224",
        "num_classes": NUM_CLASSES,
    },
    "ResNet152": {
        "path": os.path.join(WEIGHTS_DIR, "resnet152.pth"),
        "architecture": "ResNet-152",
        "input_size": "224x224",
        "num_classes": NUM_CLASSES,
    },
    "DenseNet201": {
        "path": os.path.join(WEIGHTS_DIR, "densenet201.pth"),
        "architecture": "DenseNet-201",
        "input_size": "224x224",
        "num_classes": NUM_CLASSES,
    },
    "EfficientNetB3": {
        "path": os.path.join(WEIGHTS_DIR, "efficientnet_b3.pth"),
        "architecture": "EfficientNet-B3",
        "input_size": "224x224",
        "num_classes": NUM_CLASSES,
    },
}

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@st.cache_resource
def load_all_models():
    models_dict = {}
    for model_name, info in MODELS_INFO.items():
        if os.path.exists(info["path"]):
            try:
                models_dict[model_name] = load_model(
                    model_name, NUM_CLASSES, info["path"], DEVICE
                )
            except Exception as e:
                st.error(f"Error loading {model_name}: {e}")
                models_dict[model_name] = None
        else:
            models_dict[model_name] = None
    return models_dict

st.title("Tea Leaf Disease Classification with XAI")
st.markdown("An interactive tool to classify tea leaf diseases and pests, and visualize model explanations.")
st.markdown("---")

loaded_models = load_all_models()
available_models = [name for name, model in loaded_models.items() if model is not None]

if not available_models:
    st.error("No trained models found! Please place your trained `.pth` weight files in the `weights/` directory and restart the app.")
    st.stop()

st.sidebar.header("Controls")
model_choice = st.sidebar.selectbox(
    "1. Select a Model",
    options=available_models,
    help="Choose the model to use for prediction and explanation."
)

st.sidebar.subheader("Model Metadata")
if model_choice:
    info = MODELS_INFO[model_choice]
    st.sidebar.markdown(f"""
    - **Architecture**: `{info['architecture']}`
    - **Input Size**: `{info['input_size']}`
    - **Classes**: `{info['num_classes']}`
    - **Checkpoint**: `{info['path']}`
    """)

st.sidebar.subheader("2. Choose an Image")
input_source = st.sidebar.radio("Select input type:", ("Upload an Image", "Use a Sample Image"))

image_file = None
if input_source == "Upload an Image":
    uploaded_file = st.sidebar.file_uploader(
        "Upload a JPG or PNG image", type=["jpg", "jpeg", "png"]
    )
    if uploaded_file:
        image_file = uploaded_file
else:
    sample_images = [f for f in os.listdir(SAMPLES_DIR) if f.endswith(('jpg', 'png', 'jpeg'))]
    if sample_images:
        selected_sample = st.sidebar.selectbox("Select a sample", sample_images)
        image_file = os.path.join(SAMPLES_DIR, selected_sample)
    else:
        st.sidebar.warning("No sample images found in the `sample_images` directory.")

if image_file:
    original_image = Image.open(image_file).convert("RGB")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Input Image")
        st.image(original_image, caption="Uploaded Image", use_column_width=True)

    input_tensor = TRANSFORM(original_image).unsqueeze(0).to(DEVICE)
    model = loaded_models[model_choice]
    
    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
    
    with col2:
        st.subheader("Prediction Results")
        top_indices = np.argsort(probabilities)[::-1][:3]
        
        predicted_class = CLASS_NAMES[top_indices[0]]
        confidence = probabilities[top_indices[0]]
        st.success(f"**Predicted Label:** {predicted_class} ({confidence*100:.2f}%)")
        
        st.write("**Top-3 Predictions:**")
        for i in top_indices:
            st.write(f"- {CLASS_NAMES[i]}: `{probabilities[i]*100:.2f}%`")

    st.markdown("---")
    
    st.header("Model Explanations (XAI)")
    
    if st.button("Generate All Explanations", key="generate_xai"):
        with st.spinner("Generating CAM and LIME explanations... Please wait, this may take a moment."):
            cam_visualizations = generate_cam_visualizations(model, model_choice, input_tensor, original_image)
            lime_visualization = generate_lime_visualization(model, original_image, DEVICE)

            st.subheader("Explanation Grid")
            cols = st.columns(5)
            
            all_viz = {**cam_visualizations, "LIME": lime_visualization}
            
            for idx, (name, img) in enumerate(all_viz.items()):
                with cols[idx]:
                    st.image(img, caption=name, use_column_width=True)
            
            st.subheader("Export Results")
            st.markdown("Download all generated visualizations as a single ZIP file.")
            
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                for name, img_array in all_viz.items():
                    if img_array.dtype == np.float64 or img_array.dtype == np.float32:
                         img_array = (img_array * 255).astype(np.uint8)
                    img = Image.fromarray(img_array)
                    
                    buf = io.BytesIO()
                    img.save(buf, format="PNG")
                    zf.writestr(f"{name.lower().replace(' ', '_')}.png", buf.getvalue())

            st.download_button(
                label="Download Visualizations (.zip)",
                data=zip_buffer.getvalue(),
                file_name=f"xai_visualizations_{model_choice}_tea.zip",
                mime="application/zip",
            )
else:
    st.info("Please select an image from the sidebar to begin.")