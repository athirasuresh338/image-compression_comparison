import numpy as np
from skimage.metrics import structural_similarity


# ---------------------------------------------------------------------
# Quality Metrics
# ---------------------------------------------------------------------

def calculate_psnr(original_image, compressed_image):
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR) in dB between two images.
    Higher values indicate better reconstruction fidelity.
    """
    original = original_image.astype(float)
    compressed = compressed_image.astype(float)

    # Compute Mean Squared Error
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return float("inf")

    # Standard formula for 8-bit dynamic range [0, 255]
    max_pixel = 255.0
    return 20 * np.log10(max_pixel) - 10 * np.log10(mse)


def calculate_ssim(original_image, compressed_image):
    """
    Calculate Structural Similarity Index (SSIM) for an RGB image.
    Values range from -1 to 1 (1 indicates identical perceptual structure).
    """
    min_side = min(original_image.shape[0], original_image.shape[1])
    if min_side < 3:
        return 1.0 if np.array_equal(original_image, compressed_image) else 0.0

    win_size = min(7, min_side)
    if win_size % 2 == 0:
        win_size -= 1

    return float(
        structural_similarity(
            original_image,
            compressed_image,
            channel_axis=2,
            data_range=255,
            win_size=win_size
        )
    )


# ---------------------------------------------------------------------
# Quality Interpretations
# ---------------------------------------------------------------------

def get_psnr_quality(psnr):
    """Return a descriptive rating based on standard PSNR thresholds."""
    if psnr >= 40:
        return "Excellent - Very little visual quality loss."
    elif psnr >= 30:
        return "Good - Most fine visual details are well preserved."
    elif psnr >= 20:
        return "Moderate - Noticeable compression artifacts present."
    else:
        return "Poor - Significant visual degradation."


def get_ssim_interpretation(ssim):
    """Return a descriptive rating based on standard SSIM ranges."""
    if ssim >= 0.95:
        return "Almost Identical to the original image."
    elif ssim >= 0.85:
        return "Very Similar with minimal structural differences."
    elif ssim >= 0.70:
        return "Noticeably Different from the original image."
    else:
        return "Significantly Different with noticeable structural loss."