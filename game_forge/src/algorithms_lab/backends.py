"""Optional backends and acceleration helpers."""

from __future__ import annotations

from typing import Callable, Protocol


class JitFn(Protocol):
    def __call__(self, *args, **kwargs): ...


try:  # pragma: no cover - exercised via optional dependency tests
    import numba as _numba

    HAS_NUMBA = True

    def njit(*args, **kwargs) -> Callable:
        return _numba.njit(*args, **kwargs)

    prange = _numba.prange
except Exception:  # pragma: no cover
    HAS_NUMBA = False

    def njit(*args, **kwargs) -> Callable:  # type: ignore
        raise ImportError("numba is not installed")

    def prange(*args, **kwargs):  # type: ignore
        raise ImportError("numba is not installed")


try:  # pragma: no cover - exercised via optional dependency tests
    from scipy.spatial import cKDTree as _cKDTree

    HAS_SCIPY = True
    cKDTree = _cKDTree
except Exception:  # pragma: no cover
    HAS_SCIPY = False
    cKDTree = None
