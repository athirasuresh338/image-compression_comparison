# Image Compression Comparison

An interactive Streamlit application for exploring and comparing image compression techniques using **K-Means Clustering**, **Principal Component Analysis (PCA)**, and **Singular Value Decomposition (SVD)**.

The application allows users to upload an image, adjust compression parameters, visualize compression results, and evaluate image quality using quantitative metrics.

## Live Demo

🔗 *Live demo link will be available once deployed to Streamlit Community Cloud.*

## Features

* Upload and compress images using K-Means, PCA, and SVD
* Adjustable compression parameters (clusters, components, rank)
* Side-by-side comparison of original and compressed images
* Theoretical compression ratio analysis
* PSNR (Peak Signal-to-Noise Ratio) evaluation
* SSIM (Structural Similarity Index) evaluation
* Clean, responsive Streamlit interface

## Compression Techniques

### K-Means Clustering
Reduces the number of colors in an image by grouping similar pixels into clusters and replacing them with cluster centroids.

### Principal Component Analysis (PCA)
Compresses images through dimensionality reduction while preserving the principal axes of color variance.

### Singular Value Decomposition (SVD)
Approximates images using lower-rank matrix representations to reduce the number of required singular values.

## Evaluation Metrics

### Compression Ratio
Measures the theoretical reduction in information representation achieved by the selected compression algorithm.

### PSNR (Peak Signal-to-Noise Ratio)
Measures reconstruction quality between the original and compressed image.

| PSNR Value | Quality |
| :--- | :--- |
| > 40 dB | Excellent |
| 30 – 40 dB | Good |
| 20 – 30 dB | Moderate |
| < 20 dB | Poor |

### SSIM (Structural Similarity Index)
Measures structural similarity between the original and compressed image.

| SSIM Value | Similarity |
| :--- | :--- |
| 0.95 – 1.00 | Almost Identical |
| 0.85 – 0.95 | Very Similar |
| 0.70 – 0.85 | Noticeably Different |
| < 0.70 | Significantly Different |

## Technologies Used

* Python
* Streamlit
* NumPy
* Pillow (PIL)
* Scikit-Learn
* Scikit-Image

## Project Structure

```text
image-compression_comparison/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
└── src/
    ├── __init__.py
    ├── compression.py
    └── metrics.py
```

## Run Locally

```bash
git clone https://github.com/athirasuresh338/image-compression_comparison.git
cd image-compression_comparison
pip install -r requirements.txt
streamlit run app.py
```
