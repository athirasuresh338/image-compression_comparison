import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA


# ---------------------------------------------------------------------
# K-Means Image Compression
# ---------------------------------------------------------------------

def compress_kmeans(image_array, k):
    """
    Compresses an image using K-Means color quantization.
    Reduces the color palette to k representative colors.
    """
    height, width, channels = image_array.shape

    # Flatten image to a 2D array of RGB pixels (num_pixels x channels)
    pixels = image_array.reshape(-1, channels)

    # Cluster pixels into k color centroids (safeguard against images with fewer pixels than k)
    actual_k = min(k, len(pixels))
    kmeans = MiniBatchKMeans(n_clusters=actual_k, random_state=42, batch_size=2048)
    labels = kmeans.fit_predict(pixels)

    # Replace each pixel with its assigned centroid color
    quantized_pixels = kmeans.cluster_centers_[labels]

    # Reshape back to original image dimensions
    compressed_image = quantized_pixels.reshape(height, width, channels)
    return compressed_image.astype(np.uint8)


def calculate_kmeans_ratio(image_array, k):
    """
    Calculates theoretical compression ratio for K-Means:
    Original size vs. storing the k-color palette + 1 byte index per pixel.
    """
    height, width, channels = image_array.shape

    original_size = height * width * channels
    compressed_size = (k * channels) + (height * width)

    return original_size / compressed_size if compressed_size > 0 else 1.0


# ---------------------------------------------------------------------
# PCA Image Compression
# ---------------------------------------------------------------------

def _compress_channel_pca(channel, n_components):
    """Compresses a single 2D color channel matrix using PCA."""
    max_components = min(channel.shape)
    k = min(n_components, max_components)

    # Reduce dimensions and reconstruct
    pca = PCA(n_components=k)
    transformed = pca.fit_transform(channel)
    reconstructed = pca.inverse_transform(transformed)

    return reconstructed


def compress_pca(image_array, n_components):
    """
    Compresses an image by applying PCA across each color channel (R, G, B).
    """
    compressed_channels = []

    # Process each color channel individually
    for channel_idx in range(3):
        channel = image_array[:, :, channel_idx]
        compressed_ch = _compress_channel_pca(channel, n_components)
        compressed_channels.append(compressed_ch)

    # Recombine channels back into an RGB image
    reconstructed = np.stack(compressed_channels, axis=2)

    # Clip values to ensure valid pixel intensities [0, 255]
    return np.clip(reconstructed, 0, 255).astype(np.uint8)


def calculate_pca_ratio(image_array, n_components):
    """
    Calculates theoretical compression ratio for PCA based on stored
    principal component vectors and projection coefficients per channel.
    """
    height, width, channels = image_array.shape
    k = min(n_components, min(height, width))

    original_size = height * width * channels
    compressed_per_channel = (height * k) + (k * width)
    total_compressed_size = channels * compressed_per_channel

    return original_size / total_compressed_size if total_compressed_size > 0 else 1.0


# ---------------------------------------------------------------------
# SVD Image Compression
# ---------------------------------------------------------------------

def _compress_channel_svd(channel, rank):
    """Compresses a single 2D color channel matrix using truncated SVD."""
    max_rank = min(channel.shape)
    r = min(rank, max_rank)

    # Compute full Singular Value Decomposition
    U, S, VT = np.linalg.svd(channel, full_matrices=False)

    # Truncate to top-r singular values and vectors
    U_reduced = U[:, :r]
    S_reduced = S[:r]
    VT_reduced = VT[:r, :]

    # Low-rank matrix approximation: A_r = U_r * S_r * V_r^T
    return U_reduced @ np.diag(S_reduced) @ VT_reduced


def compress_svd(image_array, rank):
    """
    Compresses an image by retaining only top-rank singular values per channel.
    """
    compressed_channels = []

    # Process each color channel individually
    for channel_idx in range(3):
        channel = image_array[:, :, channel_idx]
        compressed_ch = _compress_channel_svd(channel, rank)
        compressed_channels.append(compressed_ch)

    # Recombine channels back into an RGB image
    reconstructed = np.stack(compressed_channels, axis=2)

    # Clip values to ensure valid pixel intensities [0, 255]
    return np.clip(reconstructed, 0, 255).astype(np.uint8)


def calculate_svd_ratio(image_array, rank):
    """
    Calculates theoretical compression ratio for SVD based on stored
    singular values and truncated U and V^T matrices per channel.
    """
    height, width, channels = image_array.shape
    r = min(rank, min(height, width))

    original_size = height * width * channels
    compressed_per_channel = (height * r) + r + (r * width)
    total_compressed_size = channels * compressed_per_channel

    return original_size / total_compressed_size if total_compressed_size > 0 else 1.0
