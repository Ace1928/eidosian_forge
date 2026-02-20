"""
Lightweight Numba compatibility shims.

Termux and constrained CI environments may not provide a working numba wheel.
These fallbacks keep physics modules importable while preserving function
semantics (without JIT acceleration).
"""

from __future__ import annotations

from typing import Any, Callable

try:  # pragma: no cover - exercised implicitly where numba is available
    from numba import njit, prange  # type: ignore
except Exception:  # pragma: no cover - fallback path used on Termux/CI

    def njit(*args: Any, **kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """No-op decorator fallback when numba is unavailable."""

        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            return args[0]

        def _decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            return func

        return _decorator

    def prange(*args: int) -> range:
        """Fallback parallel range implementation."""
        return range(*args)

