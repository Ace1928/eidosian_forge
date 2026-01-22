import copy
from functools import lru_cache
from weakref import WeakValueDictionary
import numpy as np
import matplotlib as mpl
from . import _api, _path
from .cbook import _to_unmasked_float_array, simple_linear_interpolation
from .bezier import BezierSegment
def iter_bezier(self, **kwargs):
    """
        Iterate over each Bézier curve (lines included) in a `Path`.

        Parameters
        ----------
        **kwargs
            Forwarded to `.iter_segments`.

        Yields
        ------
        B : `~matplotlib.bezier.BezierSegment`
            The Bézier curves that make up the current path. Note in particular
            that freestanding points are Bézier curves of order 0, and lines
            are Bézier curves of order 1 (with two control points).
        code : `~matplotlib.path.Path.code_type`
            The code describing what kind of curve is being returned.
            `MOVETO`, `LINETO`, `CURVE3`, and `CURVE4` correspond to
            Bézier curves with 1, 2, 3, and 4 control points (respectively).
            `CLOSEPOLY` is a `LINETO` with the control points correctly
            chosen based on the start/end points of the current stroke.
        """
    first_vert = None
    prev_vert = None
    for verts, code in self.iter_segments(**kwargs):
        if first_vert is None:
            if code != Path.MOVETO:
                raise ValueError('Malformed path, must start with MOVETO.')
        if code == Path.MOVETO:
            first_vert = verts
            yield (BezierSegment(np.array([first_vert])), code)
        elif code == Path.LINETO:
            yield (BezierSegment(np.array([prev_vert, verts])), code)
        elif code == Path.CURVE3:
            yield (BezierSegment(np.array([prev_vert, verts[:2], verts[2:]])), code)
        elif code == Path.CURVE4:
            yield (BezierSegment(np.array([prev_vert, verts[:2], verts[2:4], verts[4:]])), code)
        elif code == Path.CLOSEPOLY:
            yield (BezierSegment(np.array([prev_vert, first_vert])), code)
        elif code == Path.STOP:
            return
        else:
            raise ValueError(f'Invalid Path.code_type: {code}')
        prev_vert = verts[-2:]