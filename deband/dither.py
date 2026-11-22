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

``__all__`` lists the eight public primitives.  Every one of them is currently
a stub and raises ``NotImplementedError``; the contracts below define the
behaviour the red tests in ``tests/test_dither.py`` expect.
"""

from __future__ import annotations

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


def quantize(x: float, bits: int) -> float:
    """Quantise one linear-light sample to ``bits`` bit depth.

    ``x`` is a linear-light float; values outside ``[0.0, 1.0]`` are clipped
    to the nearest endpoint first.  The sample is rounded to the **nearest**
    representable level ``k / (2 ** bits - 1)`` (half-way values round up),
    never truncated with ``floor``: the maximum error is ``0.5 LSB + 1e-9``
    and the mean error over a uniform sweep is approximately zero (a floor
    implementation is biased darker by half an LSB).  The returned float lies
    on the quantisation grid.

    ``bits`` must be an integer in ``1..16``; otherwise raise ``ValueError``.
    In particular ``bits <= 0`` must be rejected instead of dividing by
    ``2 ** bits - 1``.
    """
    raise NotImplementedError


def tpdf_noise(seed: int, n: int) -> list[float]:
    """Return ``n`` samples of 1-LSB triangular (TPDF) dither noise.

    Samples are floats in ``[-1.0, 1.0]`` measured in LSB: the standard
    triangular distribution built as the difference of two independent uniform
    values (peak-to-peak 2 LSB, standard deviation ``1 / sqrt(6) ~ 0.408``).
    The amplitude is exactly 1 LSB, never 2 LSB.

    The distribution must be triangular, not uniform: samples near zero are
    much more likely than samples near the extremes.  The mean is close to
    zero.  ``seed`` and ``n`` are integers with ``n >= 0``.  The output is
    fully determined by them (same arguments reproduce the same values within
    ``1e-9``), different seeds select different streams, and no randomness
    source may be used.
    """
    raise NotImplementedError


def blue_noise_1d(seed: int, n: int, strength: float) -> list[float]:
    """Return ``n`` samples of one-dimensional blue (high-frequency) noise.

    ``strength`` is a scale in ``[0.0, 1.0]``; every sample lies in
    ``[-strength, strength]`` and the peak approaches ``strength``.  Adjacent
    samples are **negatively correlated**: the normalised lag-1
    autocorrelation is ``r1 <= -0.25``.  White noise, whose ``r1`` is about
    zero, is not blue noise.  The mean is close to zero.

    The result is deterministic in ``(seed, n, strength)`` within ``1e-9``,
    different seeds select different streams, and no randomness source may be
    used.
    """
    raise NotImplementedError


def apply_dither(img: list[float], bits: int, strength: float, seed: int) -> list[float]:
    """Add dither noise and only then quantise, returning a float image.

    ``img`` is a flat sequence of linear-light samples in ``[0.0, 1.0]`` and
    ``bits`` is the target bit depth (``1..16``).  ``strength`` is expressed
    in LSB; ``1.0`` means ``+/-1`` LSB of TPDF noise.

    The order is mandatory: add noise **before** rounding to the
    ``1 / (2 ** bits - 1)`` grid.  Every output sample is therefore exactly on
    the quantisation grid (within ``1e-9``) and inside ``[0.0, 1.0]``.  Dither
    must genuinely break flat bands (``band_variance`` rises above the
    undithered value) while the mean absolute error stays below 1 LSB.

    The noise is derived from ``seed``: different seeds must produce different
    grain, so a static shot crawls from frame to frame, while the same seed
    reproduces the same result within ``1e-9``.  A fixed internal seed is not
    acceptable.  ``bits < 1`` raises ``ValueError``.
    """
    raise NotImplementedError


def deband_gradient(img: list[float], steps: int) -> list[float]:
    """Simulate hard colour banding by quantising to ``steps`` levels.

    ``steps`` is the **number of discrete levels** in ``[0.0, 1.0]`` (an
    integer ``>= 2``), not a bit depth and not an exponent: the levels are
    ``k / (steps - 1)``, so a ramp spanning the full range yields exactly
    ``steps`` distinct output values.  Using ``2 ** steps`` levels is the
    classic bug that over-simulates banding.

    Samples are rounded to the nearest level, clipped to ``[0.0, 1.0]`` and
    returned as floats (tolerance ``1e-9``).  ``steps < 2`` raises
    ``ValueError``.
    """
    raise NotImplementedError


def band_variance(img: list[float]) -> float:
    """Estimate the within-band variance of a flat linear-light image.

    The image is high-passed with a three-tap median: for every interior
    sample compute ``d = img[i] - median(img[i-1], img[i], img[i+1])`` and
    return ``mean(d * d)``.  The median makes the measure insensitive to step
    edges, ramps and static content, so it does **not** degrade into the
    whole-image variance, which is dominated by edges and ramps and reports
    bogus values.

    Hard, undithered bands (flat plateaus, clean steps) give ``0.0`` within
    ``1e-9``; a dithered flat region gives a clearly positive value.
    ``len(img) < 3`` returns ``0.0``.  The function is deterministic.
    """
    raise NotImplementedError


def temporal_stability(frames: list[list[float]], seed: int) -> float:
    """Measure how static the grain is across frames, in ``[0.0, 1.0]``.

    ``frames`` holds at least two equal-length float frames (typically dithered
    ``[0.0, 1.0]`` images; any finite floats are accepted).  The per-pixel
    temporal mean is subtracted first, so static scene content (ramps, edges)
    cancels and only temporal noise remains; skipping this step lets shared
    content dominate and drives the result to 1.

    The normalised correlation of adjacent residual pairs is averaged and
    clipped to ``[0.0, 1.0]`` (negative correlation becomes 0).  Frames that
    are all identical have zero residual and return ``1.0``; independent
    per-frame grain returns about ``0.0``.

    ``seed`` only selects a deterministic subsampling offset; no randomness is
    allowed.  Values are compared with tolerance ``1e-9``.  Mismatched frame
    lengths or fewer than two frames raise ``ValueError``.
    """
    raise NotImplementedError


def validate(img: list[float], bits: int) -> list[str]:
    """Return a list of human-readable problems; ``[]`` means valid.

    ``img`` must be a non-empty flat sequence of at most ``2 ** 22`` finite
    linear-light samples, each in ``[0.0, 1.0]`` within ``1e-9``.  ``bits``
    must be an integer in ``1..16``.  ``bits <= 0``, ``bits > 16``,
    non-integer bits, an empty image, an oversized image, a non-finite sample,
    or an out-of-range sample each add one string.

    Problems are reported as strings instead of raised, and the helper must
    never divide by ``2 ** bits - 1`` before checking ``bits``: ``bits == 0``
    would otherwise be a zero-division crash.
    """
    raise NotImplementedError
