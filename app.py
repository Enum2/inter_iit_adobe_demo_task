# app_sam_interactive.py
# pip install streamlit pillow numpy opencv-python torch git+https://github.com/facebookresearch/segment-anything.git gdown

import streamlit as st
import numpy as np
import cv2
import os
import tempfile
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor
import gdown

# ----------------------------------
# Helper functions
# ----------------------------------
def overlay_mask(image, mask, alpha=0.5):
    color = np.random.randint(0, 255, (3,), dtype=np.uint8)
    color_mask = np.zeros_like(image)
    color_mask[mask] = color
    overlay = cv2.addWeighted(image, 1 - alpha, color_mask, alpha, 0)
    return Image.fromarray(overlay)

def download_checkpoint(url, path):
    if not os.path.exists(path):
        st.info(f"Downloading SAM checkpoint (~358MB) to {path} ...")
        gdown.download(url, path, quiet=False)
        st.success("✅ Download complete!")

# ----------------------------------
# Streamlit UI
# ----------------------------------
st.set_page_config(page_title="Segment Anything Interactive", layout="wide")
st.title("🎨 Segment Anything — Interactive Embedding Mode")

# Checkpoint URL (set your hosted URL here, e.g., Google Drive or Hugging Face)
checkpoint_url = "https://huggingface.co/facebook/sam-vit-b/resolve/main/sam_vit_b_01ec64.pth"
checkpoint_path = "sam_vit_b_01ec64.pth"

# Download checkpoint if missing
download_checkpoint(checkpoint_url, checkpoint_path)

opacity = st.slider("Mask Opacity", 0.0, 1.0, 0.5)
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
st.write("After upload, go down to compute embeddings.")

if "predictor" not in st.session_state:
    st.session_state.predictor = None
if "image" not in st.session_state:
    st.session_state.image = None

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(img)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    if st.button("🔍 Compute Image Embeddings"):
        if not os.path.exists(checkpoint_path):
            st.error(f"Checkpoint not found at {checkpoint_path}")
            st.stop()

        with st.spinner("Loading SAM and computing embeddings..."):
            sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
            predictor = SamPredictor(sam)
            predictor.set_image(img_np)
            st.session_state.predictor = predictor
            st.session_state.image = img_np
        st.success("✅ Image embeddings computed and cached!")

# ----------------------------------
# Click-based mask prediction
# ----------------------------------
st.write("Try coordinates within the image dimensions to segment a region.")
if st.session_state.predictor is not None:
    st.write("Now click anywhere on the image to segment that region.")

    click_x = st.number_input("Click X coordinate (pixel)", min_value=0)
    click_y = st.number_input("Click Y coordinate (pixel)", min_value=0)

    if st.button("🎯 Segment Selected Point"):
        input_point = np.array([[click_x, click_y]])
        input_label = np.array([1])

        masks, scores, _ = st.session_state.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True
        )

        best_mask = masks[np.argmax(scores)]
        overlay = overlay_mask(st.session_state.image, best_mask, alpha=opacity)
        st.image(overlay, caption="Segmented Mask Overlay", use_column_width=True)

        # Optionally save / download
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            overlay.save(tmp.name)
            with open(tmp.name, "rb") as f:
                st.download_button("📥 Download Mask", data=f, file_name="mask_overlay.png")
