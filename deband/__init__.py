"""Deterministic dithering and banding diagnostics package."""

from .dither import (
    apply_dither,
    band_variance,
    blue_noise_1d,
    deband_gradient,
    quantize,
    temporal_stability,
    tpdf_noise,
    validate,
)

__all__ = [
    "apply_dither",
    "band_variance",
    "blue_noise_1d",
    "deband_gradient",
    "quantize",
    "temporal_stability",
    "tpdf_noise",
    "validate",
]
