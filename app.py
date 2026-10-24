import streamlit as st
import numpy as np
from PIL import Image

from src.compression import (
    compress_kmeans,
    calculate_kmeans_ratio,
    compress_pca,
    calculate_pca_ratio,
    compress_svd,
    calculate_svd_ratio,
)
from src.metrics import (
    calculate_psnr,
    calculate_ssim,
    get_psnr_quality,
    get_ssim_interpretation,
)


# ---------------------------------------------------------------------
# Page Setup & Configuration
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="Image Compression Comparison",
    layout="wide"
)

# Persistent state initialization across reruns
if "compressed_image" not in st.session_state:
    st.session_state.compressed_image = None

if "results" not in st.session_state:
    st.session_state.results = None

if "last_file_name" not in st.session_state:
    st.session_state.last_file_name = None


# ---------------------------------------------------------------------
# Sidebar (Controls & Settings)
# ---------------------------------------------------------------------

with st.sidebar:
    st.header("Compression Controls")

    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"]
    )

    # Reset cached outputs when a new image is uploaded
    if uploaded_file and st.session_state.last_file_name != uploaded_file.name:
        st.session_state.compressed_image = None
        st.session_state.results = None
        st.session_state.last_file_name = uploaded_file.name

    algorithm = st.selectbox(
        "Compression Method",
        ["K-Means", "PCA", "SVD"]
    )

    # Contextual parameter sliders with intuitive ranges
    if algorithm == "K-Means":
        parameter = st.slider(
            "Number of Colors (k)",
            min_value=2,
            max_value=64,
            value=16,
            help="Fewer colors yield higher compression but flatter tones."
        )
    elif algorithm == "PCA":
        parameter = st.slider(
            "Principal Components",
            min_value=1,
            max_value=100,
            value=25,
            help="Higher components preserve finer texture details."
        )
    else:
        parameter = st.slider(
            "Singular Value Rank",
            min_value=1,
            max_value=100,
            value=25,
            help="Higher rank preserves more structural matrix information."
        )

    # Action button with primary accent styling
    run_compression = st.button(
        "Run Compression",
        type="primary",
        use_container_width=True,
        disabled=(uploaded_file is None)
    )


# ---------------------------------------------------------------------
# Main View & Workflow
# ---------------------------------------------------------------------

st.title("Image Compression Comparison")
st.write(
    "Explore and compare image compression techniques using **K-Means Clustering**, "
    "**Principal Component Analysis (PCA)**, and **Singular Value Decomposition (SVD)**. "
    "Adjust parameters in the sidebar to observe the trade-offs between compression ratio and reconstructed quality."
)
st.divider()

if uploaded_file is None:
    st.info("👈 Upload an image in the sidebar to get started.")
else:
    # Prepare image and ensure RGB mode with reasonable display dimensions
    img = Image.open(uploaded_file).convert("RGB")
    img.thumbnail((800, 800))
    image_array = np.array(img)

    # Execute compression when clicked
    if run_compression:
        with st.spinner("Processing image compression..."):
            if algorithm == "K-Means":
                compressed_array = compress_kmeans(image_array, parameter)
                ratio = calculate_kmeans_ratio(image_array, parameter)

            elif algorithm == "PCA":
                compressed_array = compress_pca(image_array, parameter)
                ratio = calculate_pca_ratio(image_array, parameter)

            else:
                compressed_array = compress_svd(image_array, parameter)
                ratio = calculate_svd_ratio(image_array, parameter)

            # Compute quantitative quality metrics
            psnr = calculate_psnr(image_array, compressed_array)
            ssim = calculate_ssim(image_array, compressed_array)

            # Store results in session state
            st.session_state.compressed_image = Image.fromarray(compressed_array)
            st.session_state.results = {
                "ratio": ratio,
                "psnr": psnr,
                "ssim": ssim,
                "psnr_quality": get_psnr_quality(psnr),
                "ssim_quality": get_ssim_interpretation(ssim),
            }

    # Side-by-side comparison view
    img_col1, img_col2 = st.columns(2)

    with img_col1:
        st.subheader("Original Image")
        st.image(img, use_container_width=True)

    with img_col2:
        st.subheader("Compressed Image")
        if st.session_state.compressed_image is not None:
            st.image(st.session_state.compressed_image, use_container_width=True)
        else:
            st.info("Click **Run Compression** in the sidebar to generate the result.")

    # Results display
    if st.session_state.results is not None:
        res = st.session_state.results

        st.divider()
        st.subheader("Compression Results")

        m1, m2, m3 = st.columns(3)

        with m1:
            st.metric("Compression Ratio", f"{res['ratio']:.2f}x")
            st.caption(f"Requires ~{res['ratio']:.2f}× less storage than the raw pixel array.")

        with m2:
            st.metric("PSNR", f"{res['psnr']:.2f} dB")
            st.caption(res["psnr_quality"])

        with m3:
            st.metric("SSIM", f"{res['ssim']:.4f}")
            st.caption(res["ssim_quality"])
