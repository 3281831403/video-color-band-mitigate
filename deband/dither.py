"""Deterministic dithering and banding diagnostics for linear-light images.

Images are flat, row-major ``list[float]`` sequences of **linear-light**
samples in ``[0.0, 1.0]`` (never sRGB-encoded values).  Quantisation uses a
bit depth ``bits`` with ``1 <= bits <= 16``; one LSB is
``1 / (2 ** bits - 1)``.  Outputs stay floating point so callers can keep
processing in linear light.

The module is offline and deterministic: noise is derived from an explicit
integer ``seed`` through a stable hash, never from the ``random`` module, the
clock, or any process-global state.  Repeating a call with identical arguments
must reproduce identical values.  Numeric results are compared with an
absolute tolerance of ``1e-9``.

``__all__`` lists the eight public primitives.
"""

from __future__ import annotations

import math

__all__ = [
    "quantize",
    "tpdf_noise",
    "blue_noise_1d",
    "apply_dither",
    "deband_gradient",
    "band_variance",
    "temporal_stability",
    "validate",
]

_MASK64 = 0xFFFFFFFFFFFFFFFF


def _hash01(seed: int, index: int, stream: int = 0) -> float:
    """Deterministic uniform in ``[0.0, 1.0)`` from (seed, index, stream).

    Uses the splitmix64 finaliser over a mixed key; no global state, no
    ``random`` module, fully reproducible across runs and platforms.
    """
    z = (
        (seed & _MASK64) * 0x9E3779B97F4A7C15
        + (index & _MASK64) * 0xBF58476D1CE4E5B9
        + (stream & _MASK64) * 0x94D049BB133111EB
        + 0x2545F4914F6CDD1D
    ) & _MASK64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
    z = z ^ (z >> 31)
    return z / 18446744073709551616.0  # 2 ** 64


def _check_bits(bits: int) -> None:
    if isinstance(bits, bool) or not isinstance(bits, int):
        raise ValueError(f"bits must be an integer in 1..16, got {bits!r}")
    if bits < 1 or bits > 16:
        raise ValueError(f"bits must be in 1..16, got {bits}")


def quantize(x: float, bits: int) -> float:
    """Quantise one linear-light sample to ``bits`` bit depth (nearest level)."""
    _check_bits(bits)
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    levels = (1 << bits) - 1
    k = math.floor(x * levels + 0.5)  # round half up, never floor
    if k < 0:
        k = 0
    elif k > levels:
        k = levels
    return k / levels


def tpdf_noise(seed: int, n: int) -> list[float]:
    """Return ``n`` samples of 1-LSB triangular (TPDF) dither noise in [-1, 1]."""
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    out = []
    for i in range(n):
        u1 = _hash01(seed, i, 0)
        u2 = _hash01(seed, i, 1)
        out.append(u1 - u2)  # triangular on (-1, 1), std 1/sqrt(6), peak 1 LSB
    return out


def blue_noise_1d(seed: int, n: int, strength: float) -> list[float]:
    """Return ``n`` samples of 1-D blue (high-frequency) noise.

    Built as the first difference of white noise, which gives a lag-1
    autocorrelation of -0.5 and a peak approaching ``strength``.
    """
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    white = [_hash01(seed, i, 2) * 2.0 - 1.0 for i in range(n + 1)]
    return [
        strength * 0.5 * (white[i + 1] - white[i]) for i in range(n)
    ]


def apply_dither(img: list[float], bits: int, strength: float, seed: int) -> list[float]:
    """Add TPDF dither noise and only then quantise, returning a float image."""
    _check_bits(bits)
    lsb = 1.0 / ((1 << bits) - 1)
    noise = tpdf_noise(seed, len(img))
    amp = strength * lsb
    return [quantize(x + amp * noise[i], bits) for i, x in enumerate(img)]


def deband_gradient(img: list[float], steps: int) -> list[float]:
    """Simulate hard colour banding by quantising to ``steps`` plain levels."""
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 2:
        raise ValueError(f"steps must be an integer >= 2, got {steps!r}")
    levels = steps - 1
    out = []
    for x in img:
        if x <= 0.0:
            out.append(0.0)
        elif x >= 1.0:
            out.append(1.0)
        else:
            out.append(math.floor(x * levels + 0.5) / levels)
    return out


def band_variance(img: list[float]) -> float:
    """Within-band variance via a three-tap median high-pass."""
    n = len(img)
    if n < 3:
        return 0.0
    total = 0.0
    count = 0
    for i in range(1, n - 1):
        a, b, c = img[i - 1], img[i], img[i + 1]
        med = a + b + c - min(a, b, c) - max(a, b, c)
        d = b - med
        total += d * d
        count += 1
    return total / count


def temporal_stability(frames: list[list[float]], seed: int) -> float:
    """Correlation of temporal noise across frames, clipped to [0.0, 1.0]."""
    if len(frames) < 2:
        raise ValueError("need at least two frames")
    n = len(frames[0])
    for frame in frames:
        if len(frame) != n:
            raise ValueError("all frames must have equal length")
    if n == 0:
        return 1.0

    # Deterministic subsampling offset selected by the seed.
    start = seed % n
    idx = range(start, n)
    count = n - start

    # Per-pixel temporal mean, so static scene content cancels.
    means = [0.0] * count
    for frame in frames:
        for j, i in enumerate(idx):
            means[j] += frame[i]
    inv = 1.0 / len(frames)
    means = [m * inv for m in means]

    residuals = [
        [frame[i] - means[j] for j, i in enumerate(idx)] for frame in frames
    ]

    corrs = []
    for f in range(len(frames) - 1):
        a = residuals[f]
        b = residuals[f + 1]
        mean_a = sum(a) / count
        mean_b = sum(b) / count
        var_a = sum((v - mean_a) ** 2 for v in a)
        var_b = sum((v - mean_b) ** 2 for v in b)
        if var_a <= 0.0 and var_b <= 0.0:
            corrs.append(1.0)  # no temporal noise at all: perfectly stable
        elif var_a <= 0.0 or var_b <= 0.0:
            corrs.append(0.0)
        else:
            cov = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
            corrs.append(cov / math.sqrt(var_a * var_b))

    result = sum(corrs) / len(corrs)
    return min(1.0, max(0.0, result))


def validate(img: list[float], bits: int) -> list[str]:
    """Return a list of human-readable problems; ``[]`` means valid."""
    problems: list[str] = []

    # Check bits before anything that could divide by 2 ** bits - 1.
    if isinstance(bits, bool) or not isinstance(bits, int):
        problems.append(f"bits must be an integer, got {bits!r}")
    elif bits < 1:
        problems.append(f"bits must be >= 1, got {bits}")
    elif bits > 16:
        problems.append(f"bits must be <= 16, got {bits}")

    if len(img) == 0:
        problems.append("image is empty")
    elif len(img) > 2 ** 22:
        problems.append(f"image too large: {len(img)} samples > 2 ** 22")

    if any(not math.isfinite(v) for v in img):
        problems.append("image contains a non-finite sample")
    if any(v < -1e-9 or v > 1.0 + 1e-9 for v in img):
        problems.append("image contains a sample outside [0.0, 1.0]")

    return problems
