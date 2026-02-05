import numpy as np

from algorithms_lab.core import Domain, WrapMode, axis_aligned_bounds


def test_domain_wrap_and_clamp() -> None:
    domain = Domain(mins=np.array([0.0, 0.0]), maxs=np.array([1.0, 1.0]), wrap=WrapMode.WRAP)
    pts = np.array([[1.2, -0.1], [-0.3, 0.5]], dtype=np.float32)
    wrapped = domain.wrap_positions(pts)
    assert np.all((wrapped >= 0.0) & (wrapped <= 1.0))

    clamped = Domain(mins=np.array([0.0, 0.0]), maxs=np.array([1.0, 1.0]), wrap=WrapMode.CLAMP)
    clamped_pts = clamped.apply_boundary(pts)
    assert np.all((clamped_pts >= 0.0) & (clamped_pts <= 1.0))


def test_minimal_image() -> None:
    domain = Domain(mins=np.array([0.0, 0.0]), maxs=np.array([1.0, 1.0]), wrap=WrapMode.WRAP)
    delta = np.array([[0.6, 0.0], [-0.6, 0.0]], dtype=np.float32)
    adjusted = domain.minimal_image(delta)
    assert np.all(np.abs(adjusted[:, 0]) <= 0.5 + 1e-6)


def test_axis_aligned_bounds() -> None:
    pts = np.array([[0.0, 1.0], [2.0, -1.0]], dtype=np.float32)
    mins, maxs = axis_aligned_bounds(pts, padding=0.5)
    assert np.allclose(mins, [-0.5, -1.5])
    assert np.allclose(maxs, [2.5, 1.5])
