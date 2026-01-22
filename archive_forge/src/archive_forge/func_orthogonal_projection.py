from __future__ import annotations
import math as _math
import typing as _typing
import warnings as _warnings
from operator import mul as _mul
from collections.abc import Iterable as _Iterable
from collections.abc import Iterator as _Iterator
@classmethod
def orthogonal_projection(cls: type[Mat4T], left: float, right: float, bottom: float, top: float, z_near: float, z_far: float) -> Mat4T:
    """Create a Mat4 orthographic projection matrix for use with OpenGL.

        Given left, right, bottom, top values, and near/far z planes,
        create a 4x4 Projection Matrix. This is useful for setting
        :py:attr:`~pyglet.window.Window.projection`.
        """
    width = right - left
    height = top - bottom
    depth = z_far - z_near
    sx = 2.0 / width
    sy = 2.0 / height
    sz = 2.0 / -depth
    tx = -(right + left) / width
    ty = -(top + bottom) / height
    tz = -(z_far + z_near) / depth
    return cls((sx, 0.0, 0.0, 0.0, 0.0, sy, 0.0, 0.0, 0.0, 0.0, sz, 0.0, tx, ty, tz, 1.0))