import streamlit as st
import torch
import torchvision
import torch.nn as nn
import numpy as np
from PIL import Image
import cv2
import plotly.graph_objects as go
import os
from streamlit_drawable_canvas import st_canvas
import matplotlib.pyplot as plt

# Set page configuration for wider layout
st.set_page_config(layout="wide")

st.markdown("""
   <style>
    .logo-container {
        position: absolute;
        top: 20px;
        right: 40px;
        z-index: 100;
    }
    .logo-img {
        height: 60px;
    }
    </style>
    <div class="logo-container">
        <img class="logo-img" src="https://nelissenbv.nl/wp-content/uploads/2023/01/cropped-Nelissen-favicon-1.png" alt="Logo">
    </div>
""", unsafe_allow_html=True)

# ----------------------
# Configuration
# ----------------------
MODEL_PATH = "./measure_tool/deeplabv3_phase5.pth"
WINDOW_CLASS_ID = 3  # Adjust if needed
RESIZE_SIZE = (512, 512)

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torchvision.models.segmentation.deeplabv3_resnet50(weights=None, aux_loss=True)
    model.classifier[4] = nn.Conv2d(256, 12, kernel_size=(1, 1))  # Adjust if you have fewer classes
    if not os.path.isfile(MODEL_PATH):
        st.error(f"Model not found at {MODEL_PATH}")
        st.stop()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()
    return model, device

def preprocess_image(image_pil):
    # Resize to 512x512
    image_resized = image_pil.resize(RESIZE_SIZE, Image.Resampling.LANCZOS)
    image_np = np.array(image_resized)
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
    image_tensor = image_tensor.unsqueeze(0).to(device)
    return image_tensor, image_np

# ----------------------
# UI
# ----------------------
st.title("Facade Window/Door Measurement Tool")

uploaded_file = st.file_uploader("Upload facade image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image_pil = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image_pil)
    st.image(image_pil, caption=f"Uploaded Image ({image_np.shape[1]}x{image_np.shape[0]} pixels)", width=600)

    # Instructions for reference line
    st.write("Draw a blue line on the image below to mark a feature with a known length (e.g., a 100 cm door or window).")
    st.write("Tip: Ensure the line spans a feature whose real-world size you know.")
    
    # Toggle for horizontal line constraint
    constrain_horizontal = st.checkbox("Constrain to Horizontal Line", value=True)
    
    # Set canvas width to match the displayed image width, adjust height to maintain aspect ratio
    canvas_width = 600
    aspect_ratio = image_np.shape[1] / image_np.shape[0]
    canvas_height = int(canvas_width / aspect_ratio)
    
    canvas_result = st_canvas(
        stroke_width=3,
        stroke_color="blue",
        background_image=image_pil,
        drawing_mode="line",
        height=canvas_height,
        width=canvas_width,
        key="canvas",
        update_streamlit=True,
        display_toolbar=True
    )

    # Check if a line was drawn
    ref_pixel_distance = None
    x1, y1, x2, y2 = None, None, None, None
    if canvas_result.json_data is not None and len(canvas_result.json_data["objects"]) > 0:
        obj = canvas_result.json_data["objects"][0]
        if obj["type"] == "line":
            x1, y1 = obj["x1"], obj["y1"]
            x2, y2 = obj["x2"], obj["y2"]
            # Apply horizontal constraint if enabled
            if constrain_horizontal:
                y2 = y1  # Force the line to be horizontal at the starting y-coordinate
            # Calculate pixel distance on the canvas scale
            ref_pixel_distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if ref_pixel_distance < 10:
                st.warning("Reference line is too short. Please draw a longer line.")
                ref_pixel_distance = None

    # Ask for real-world length if a valid line was drawn
    if ref_pixel_distance:
        real_length_cm = st.number_input(
            "Enter real-world length of the blue reference line (in cm)",
            min_value=0.1,
            value=100.0
        )
        # Calculate pixels_per_cm based on canvas scale, adjusted for resize
        resize_factor_width = RESIZE_SIZE[0] / image_np.shape[1]
        resize_factor_height = RESIZE_SIZE[1] / image_np.shape[0]
        canvas_to_original_x = image_np.shape[1] / canvas_width
        canvas_to_original_y = image_np.shape[0] / canvas_height
        ref_pixel_distance_original = ref_pixel_distance * canvas_to_original_x  # Approximate adjustment
        pixels_per_cm = ref_pixel_distance_original / real_length_cm

        # Validate pixels_per_cm
        if pixels_per_cm < 0.1 or pixels_per_cm > 100:
            st.warning("The calculated pixels per cm ratio seems unrealistic. Please check the line or length input.")
        else:
            model, device = load_model()

            # Preprocess image for inference
            image_tensor, image_np_resized = preprocess_image(image_pil)
            orig_height, orig_width = image_np.shape[:2]

            with torch.no_grad():
                output = model(image_tensor)['out']
                pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

            # Convert prediction mask to RGB using tab20 colormap
            cmap = plt.get_cmap("tab20")
            norm = plt.Normalize(vmin=0, vmax=11)
            rgba_mask = cmap(norm(pred_mask))
            rgb_mask = (rgba_mask[:, :, :3] * 255).astype(np.uint8)  # Remove alpha channel and scale to 0-255

            # Resize prediction mask to original dimensions for display
            rgb_mask_display = cv2.resize(rgb_mask, (orig_width, orig_height), interpolation=cv2.INTER_NEAREST)

            # Display the raw prediction mask resized to original image size
            st.image(rgb_mask_display, caption=f"Raw Prediction Mask ({orig_width}x{orig_height})", channels="RGB")

            # Resize prediction mask back to original dimensions for measurement
            pred_mask_resized = cv2.resize(pred_mask, (orig_width, orig_height), interpolation=cv2.INTER_NEAREST)

            # Mask + contour
            window_mask = (pred_mask_resized == WINDOW_CLASS_ID).astype(np.uint8) * 255
            window_mask = cv2.morphologyEx(window_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            contours, _ = cv2.findContours(window_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Plot
            fig = go.Figure()

            # Add original image
            fig.add_layout_image(
                dict(
                    source=Image.fromarray(image_np),
                    xref="x", yref="y",
                    x=0, y=0,
                    sizex=orig_width,
                    sizey=orig_height,
                    sizing="stretch",
                    layer="below"
                )
            )

            # Invert y-axis and maintain equal aspect ratio
            fig.update_yaxes(autorange="reversed")
            fig.update_xaxes(scaleanchor="y")

            # Draw reference line (scaled to original coordinates)
            x1_orig = x1 * canvas_to_original_x
            x2_orig = x2 * canvas_to_original_x
            y1_orig = y1 * canvas_to_original_y
            y2_orig = y2 * canvas_to_original_y
            fig.add_trace(go.Scatter(
                x=[x1_orig, x2_orig],
                y=[y1_orig, y2_orig],
                mode='lines',
                line=dict(color="blue", width=3),
                hoverinfo="skip",
                showlegend=False
            ))

            # Add annotation for the reference line label, offset to the right
            fig.add_annotation(
                x=x2_orig + 50,  
                y=y2_orig,
                text=f"{real_length_cm} cm",
                showarrow=False,
                font=dict(color="blue", size=16),
                align="left",
                bgcolor="white",
                bordercolor="blue",
                borderwidth=1
            )

            # Draw window bounding boxes
            for i, cnt in enumerate(contours):
                x, y, w, h = cv2.boundingRect(cnt)
                width_cm = round(w / pixels_per_cm, 1)
                height_cm = round(h / pixels_per_cm, 1)
                hover_text = f"Window {i+1}<br>W: {width_cm} cm<br>H: {height_cm} cm"
                fig.add_trace(go.Scatter(
                    x=[x, x + w, x + w, x, x],
                    y=[y, y, y + h, y + h, y],
                    mode='lines',
                    fill='toself',
                    line=dict(color='limegreen', width=2),
                    text=hover_text,
                    hoverinfo='text',
                    showlegend=False
                ))

            # Lock layout to original image size and disable axis clutter
            fig.update_layout(
                width=orig_width,
                height=orig_height,
                margin=dict(l=0, r=0, t=0, b=0),
                xaxis=dict(
                    range=[0, orig_width],
                    showticklabels=False,
                    showgrid=False,
                    zeroline=False
                ),
                yaxis=dict(
                    range=[orig_height, 0],
                    showticklabels=False,
                    showgrid=False,
                    zeroline=False
                ),
                showlegend=False,
                hovermode="closest"
            )

            # Plot without resizing
            st.plotly_chart(fig, use_container_width=False)