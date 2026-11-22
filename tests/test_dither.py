"""Red tests for the linear-light dithering / banding toolkit.

The tests are deliberately read-only: fix ``deband/dither.py`` instead of
editing this file.  Every test fails with ``NotImplementedError`` raised by one
of the eight public stubs; the contracts live in the docstrings of
``deband/dither.py``.

Samples are flat ``list[float]`` images in linear light ``[0.0, 1.0]``.  One
LSB is ``1 / (2 ** bits - 1)``.  Numeric comparisons use an absolute tolerance
of ``1e-9`` unless a looser statistical bound is stated.  Noise is always
deterministic in an explicit integer ``seed``; nothing here uses ``random``.
"""

import math

import pytest

from deband.dither import (
    apply_dither,
    band_variance,
    blue_noise_1d,
    deband_gradient,
    quantize,
    temporal_stability,
    tpdf_noise,
    validate,
)

TOL = 1e-9


def _corr(a, b):
    """Normalised Pearson correlation of two equal-length float sequences."""
    assert len(a) == len(b) and len(a) > 1
    mean_a = sum(a) / len(a)
    mean_b = sum(b) / len(b)
    var_a = sum((value - mean_a) ** 2 for value in a)
    var_b = sum((value - mean_b) ** 2 for value in b)
    if var_a <= 0.0 or var_b <= 0.0:
        return 0.0
    cov = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
    return cov / math.sqrt(var_a * var_b)


def _lag1(seq):
    """Normalised lag-1 autocorrelation (biased estimator)."""
    assert len(seq) > 2
    mean = sum(seq) / len(seq)
    var = sum((value - mean) ** 2 for value in seq)
    assert var > 0.0
    cov = sum((seq[i] - mean) * (seq[i + 1] - mean) for i in range(len(seq) - 1))
    return cov / var


# ---------------------------------------------------------------------------
# One test per planted defect (10).
# ---------------------------------------------------------------------------


def test_tpdf_noise_is_triangular():
    """Defect 1: tpdf_noise must not use a uniform distribution."""
    n = 4096
    noise = tpdf_noise(123, n)

    assert len(noise) == n
    mean = sum(noise) / n
    std = math.sqrt(sum((value - mean) ** 2 for value in noise) / n)
    assert abs(mean) <= 0.05
    # TPDF has std = 1/sqrt(6) ~ 0.408; uniform on [-1, 1] has 0.577.
    assert 0.35 <= std <= 0.46

    near_zero = sum(1 for value in noise if abs(value) <= 0.25) / n
    near_edge = sum(1 for value in noise if abs(value) > 0.9) / n
    assert 0.36 <= near_zero <= 0.52  # TPDF ~0.4375; uniform 0.25
    assert near_edge <= 0.05  # TPDF ~0.01; uniform ~0.10

    repeated = tpdf_noise(123, n)
    assert max(abs(a - b) for a, b in zip(noise, repeated)) <= TOL
    other = tpdf_noise(124, n)
    assert sum(1 for a, b in zip(noise, other) if abs(a - b) > TOL) > 0


def test_tpdf_noise_amplitude_is_one_lsb():
    """Defect 2: tpdf_noise amplitude must be 1 LSB, not 2 LSB."""
    noise = tpdf_noise(5, 8192)
    peak = max(abs(value) for value in noise)

    assert peak <= 1.0 + TOL
    assert peak >= 0.9
    assert all(abs(value) <= 1.0 + TOL for value in noise)


def test_blue_noise_has_negative_lag1_autocorrelation():
    """Defect 3: blue_noise_1d must not be white noise."""
    n = 4096
    noise = blue_noise_1d(7, n, 1.0)

    assert len(noise) == n
    peak = max(abs(value) for value in noise)
    assert peak <= 1.0 + TOL
    assert peak >= 0.5
    assert abs(sum(noise) / n) <= 0.1
    # White noise sits near 0; blue noise is negatively correlated.
    assert _lag1(noise) <= -0.25

    repeated = blue_noise_1d(7, n, 1.0)
    assert max(abs(a - b) for a, b in zip(noise, repeated)) <= TOL
    other = blue_noise_1d(8, n, 1.0)
    assert sum(1 for a, b in zip(noise, other) if abs(a - b) > TOL) > 0

    scaled = blue_noise_1d(7, n, 0.5)
    assert max(abs(value) for value in scaled) <= 0.5 + TOL


def test_apply_dither_noises_before_quantizing():
    """Defect 4: dither must be added before quantisation, not after it."""
    n = 1024
    flat = [0.5] * n
    out = apply_dither(flat, 8, 1.0, 11)

    assert len(out) == n
    # Noise before rounding leaves every sample exactly on the 8-bit grid.
    for value in out:
        assert value * 255.0 == pytest.approx(round(value * 255.0), abs=1e-6)

    # An undithered flat patch is a dead band: within-band variance is zero.
    hard = [quantize(0.5, 8)] * n
    assert band_variance(hard) == pytest.approx(0.0, abs=TOL)
    # Dither has to break that band.
    assert band_variance(out) > 1e-8
    assert band_variance(out) > band_variance(hard) + 1e-9
    # Mean error stays below 1 LSB, so the patch does not drift.
    assert abs(sum(out) / n - 0.5) <= 1.0 / 255.0


def test_apply_dither_changes_grain_with_seed():
    """Defect 5: per-frame seeds must change the grain, not reuse one seed."""
    img = [0.5] * 1024
    first = apply_dither(img, 8, 1.0, 1)
    second = apply_dither(img, 8, 1.0, 2)

    changed = sum(1 for a, b in zip(first, second) if abs(a - b) > TOL)
    assert changed >= len(img) // 5
    assert abs(_corr(first, second)) <= 0.5


def test_quantize_rounds_to_nearest_level():
    """Defect 6: quantize must round to nearest, not floor half an LSB down."""
    lsb = 1.0 / 255.0
    assert quantize(0.75 * lsb, 8) == pytest.approx(lsb, abs=TOL)
    assert quantize(0.25 * lsb, 8) == pytest.approx(0.0, abs=TOL)

    samples = [i / 997.0 for i in range(1, 997)]
    mean_error = sum(quantize(value, 8) - value for value in samples) / len(samples)
    # A floor implementation is darker by about half an LSB.
    assert abs(mean_error) <= 0.1 / 255.0
    assert max(abs(quantize(value, 8) - value) for value in samples) <= 0.5 / 255.0 + TOL


def test_deband_gradient_counts_plain_levels():
    """Defect 7: steps is a level count, not an exponent of the bit depth."""
    ramp = [i / 255.0 for i in range(256)]
    out = deband_gradient(ramp, 5)

    assert len(out) == len(ramp)
    grid = [k / 4.0 for k in range(5)]
    for value in out:
        assert min(abs(value - level) for level in grid) <= TOL
    assert len({round(value * 4) for value in out}) == 5
    assert out[0] == pytest.approx(0.0, abs=TOL)
    assert out[-1] == pytest.approx(1.0, abs=TOL)

    binary = deband_gradient(ramp, 2)
    assert sorted({round(value) for value in binary}) == [0, 1]


def test_band_variance_ignores_edges_and_ramps():
    """Defect 8: band_variance must not fall back to whole-image variance."""
    step = [0.0] * 64 + [1.0] * 64
    assert band_variance(step) == pytest.approx(0.0, abs=TOL)
    assert band_variance([0.25] * 128) == pytest.approx(0.0, abs=TOL)
    assert band_variance([i / 255.0 for i in range(256)]) == pytest.approx(0.0, abs=TOL)

    lsb = 1.0 / 255.0
    rippled = [0.25 + (0.5 if i % 2 else -0.5) * lsb for i in range(256)]
    assert band_variance(rippled) > 1e-8


def test_temporal_stability_detects_independent_frames():
    """Defect 9: temporal_stability must not compare a frame with itself."""
    n = 2048
    first = tpdf_noise(31, n)
    second = tpdf_noise(32, n)
    third = tpdf_noise(33, n)

    stability = temporal_stability([first, second, third], 0)
    assert 0.0 <= stability <= 1.0
    assert stability <= 0.2
    # A perfectly static pattern is the only case that scores 1.
    assert temporal_stability([first, first, first], 0) >= 0.99


def test_validate_rejects_nonpositive_bits():
    """Defect 10: validate must reject bits <= 0 instead of dividing by zero."""
    assert validate([0.5] * 16, 8) == []

    for bits in (0, -1, 17):
        problems = validate([0.5] * 16, bits)
        assert isinstance(problems, list)
        assert problems

    assert validate([], 8) != []
    assert validate([0.5, 1.2, 0.4], 8) != []
    assert validate([0.5, -0.01, 0.4], 8) != []


# ---------------------------------------------------------------------------
# Five additional contract tests (quantisation error, band breaking,
# determinism, temporal correlation, output range).
# ---------------------------------------------------------------------------


def test_dither_16bit_mean_error_below_one_lsb():
    """A 16-bit dithered ramp keeps the average quantisation error < 1 LSB."""
    img = [i / 1023.0 for i in range(1024)]
    out = apply_dither(img, 16, 1.0, 99)
    lsb = 1.0 / 65535.0

    mean_error = sum(abs(o - x) for o, x in zip(out, img)) / len(img)
    assert mean_error < lsb
    assert max(abs(o - x) for o, x in zip(out, img)) <= 1.5 * lsb + 1e-9


def test_dither_increases_flat_region_variance():
    """Dither breaks a hard flat band: within-band variance must increase."""
    n = 512
    hard = deband_gradient([0.5] * n, 8)
    dithered = apply_dither([0.5] * n, 8, 1.0, 5)

    hard_variance = band_variance(hard)
    dithered_variance = band_variance(dithered)
    assert hard_variance == pytest.approx(0.0, abs=TOL)
    assert dithered_variance > 1e-8
    assert dithered_variance > hard_variance + 1e-9


def test_apply_dither_is_deterministic_per_seed():
    """Same frame and same seed must reproduce the same floats within 1e-9."""
    img = [i / 255.0 for i in range(256)]
    first = apply_dither(img, 8, 0.75, 42)
    again = apply_dither(img, 8, 0.75, 42)
    other = apply_dither(img, 8, 0.75, 43)

    assert len(first) == len(img)
    assert max(abs(a - b) for a, b in zip(first, again)) <= TOL
    assert sum(1 for a, b in zip(first, other) if abs(a - b) > TOL) >= 1


def test_temporal_stability_of_seeded_frames_is_low():
    """Advancing the seed per frame gives temporal correlation < 0.2."""
    img = [i / 511.0 for i in range(512)]
    frames = [apply_dither(img, 8, 1.0, 100 + k) for k in range(4)]

    stability = temporal_stability(frames, 7)
    assert 0.0 <= stability <= 1.0
    assert stability < 0.2


def test_apply_dither_keeps_output_in_range():
    """Dither output stays floating point inside [0.0, 1.0] for every bit depth."""
    img = [0.0, 0.001, 0.25, 0.5, 0.75, 0.999, 1.0] * 32

    for bits in (1, 4, 8, 12, 16):
        out = apply_dither(img, bits, 1.0, bits)
        assert len(out) == len(img)
        assert min(out) >= 0.0
        assert max(out) <= 1.0
        assert all(isinstance(value, float) for value in out)

    assert quantize(out[0], 16) == pytest.approx(out[0], abs=TOL)
