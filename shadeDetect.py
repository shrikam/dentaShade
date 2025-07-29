import streamlit as st
import numpy as np
import cv2
from PIL import Image

# Approximate RGB values of some common VITA Classical shades
VITA_SHADES = {
    "A1": (235, 219, 202),
    "A2": (232, 211, 185),
    "A3": (210, 191, 166),
    "A3.5": (193, 172, 145),
    "B1": (240, 230, 200),
    "B2": (230, 218, 180),
    "C1": (220, 205, 170),
    "D2": (210, 195, 160)
}

def closest_shade(avg_color):
    min_dist = float('inf')
    closest = None
    for shade, rgb in VITA_SHADES.items():
        dist = np.linalg.norm(np.array(avg_color) - np.array(rgb))
        if dist < min_dist:
            min_dist = dist
            closest = shade
    return closest

def main():
    st.set_page_config(page_title="Tooth Shade Detector", layout="centered")
    st.title("🦷 Tooth Shade Detection")
    st.markdown("Upload a **close-up image of teeth**, and the app will estimate the closest VITA shade based on average color.")

    uploaded_file = st.file_uploader("Choose an image (JPG or PNG)", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="📷 Uploaded Image", use_container_width=True)

        # Convert to NumPy array
        image_np = np.array(image)

        # Define and extract a central region of interest (ROI)
        h, w, _ = image_np.shape
        x1, x2 = int(w * 0.4), int(w * 0.6)
        y1, y2 = int(h * 0.4), int(h * 0.6)
        roi = image_np[y1:y2, x1:x2]

        st.markdown("### 🔍 Analyzing Central Region")
        st.image(roi, caption="Region of Interest", width=200)

        # Compute average RGB color in ROI
        avg_color = roi.mean(axis=(0, 1)).astype(int)
        predicted_shade = closest_shade(avg_color)

        st.markdown(f"### 🎯 Predicted Shade: `{predicted_shade}`")
        st.markdown(f"**Average RGB:** {tuple(avg_color)}")

        # Display shade color comparison
        st.markdown("### 🎨 Color Comparison")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Detected Average Color**")
            avg_color_image = np.ones((50, 200, 3), dtype=np.uint8)
            avg_color_image[:, :] = avg_color
            st.image(avg_color_image, use_container_width=True)
        with col2:
            st.markdown(f"**Closest VITA Shade: {predicted_shade}**")
            vita_color_image = np.ones((50, 200, 3), dtype=np.uint8)
            vita_color_image[:, :] = VITA_SHADES[predicted_shade]
            st.image(vita_color_image, use_container_width=True)

if __name__ == "__main__":
    main()
