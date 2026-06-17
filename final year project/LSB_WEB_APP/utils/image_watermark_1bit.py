"""
1-bit LSB Image Watermarking (per RGB channel)
Embeds the MSB of each watermark channel into the LSB of the corresponding
cover channel — 3 bits total per pixel, extremely imperceptible.
"""

from PIL import Image
import numpy as np
import os


def embed_extract_1bit_rgb(cover_path, watermark_path, output_dir):
    """
    Embed watermark using 1-bit LSB substitution on each RGB channel.
    The watermark is automatically resized to match the cover image.

    Returns
    -------
    tuple[str, str, str]
        Paths to: watermarked image, extracted-from-watermarked, extracted-from-cover
    """
    cover = Image.open(cover_path).convert("RGB")
    watermark = Image.open(watermark_path).convert("RGB")

    if cover.size != watermark.size:
        watermark = watermark.resize(cover.size, Image.Resampling.LANCZOS)

    cover_arr = np.array(cover)
    watermark_arr = np.array(watermark)

    h, w, c = cover_arr.shape

    # Embed 1 bit (MSB of watermark) into LSB of each cover channel
    embedded = cover_arr.copy()
    for channel in range(3):
        watermark_bits = (watermark_arr[:, :, channel] >> 7) & 0x01
        embedded[:, :, channel] = (cover_arr[:, :, channel] & 0xFE) | watermark_bits

    wm_path = os.path.join(output_dir, "watermarked_1bit_rgb.png")
    Image.fromarray(embedded.astype(np.uint8)).save(wm_path)

    # Extract from watermarked image (LSB → MSB position)
    extracted = np.zeros((h, w, c), dtype=np.uint8)
    for channel in range(3):
        extracted[:, :, channel] = (embedded[:, :, channel] & 0x01) << 7

    ext_wm_path = os.path.join(output_dir, "extracted_watermarked_1bit_rgb.png")
    Image.fromarray(extracted.astype(np.uint8)).save(ext_wm_path)

    # Extract from original cover (for comparison)
    extracted_cover = np.zeros((h, w, c), dtype=np.uint8)
    for channel in range(3):
        extracted_cover[:, :, channel] = (cover_arr[:, :, channel] & 0x01) << 7

    ext_cover_path = os.path.join(output_dir, "extracted_cover_1bit_rgb.png")
    Image.fromarray(extracted_cover.astype(np.uint8)).save(ext_cover_path)

    return wm_path, ext_wm_path, ext_cover_path
