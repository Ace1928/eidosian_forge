from . import _check_status, _keepref, cairo, constants, ffi
from .fonts import FontFace, FontOptions, ScaledFont, _encode_string
from .matrix import Matrix
from .patterns import Pattern
from .surfaces import Surface
def fill_extents(self):
    """Computes a bounding box in user-space coordinates
        covering the area that would be affected, (the "inked" area),
        by a :meth:`fill` operation given the current path and fill parameters.
        If the current path is empty,
        returns an empty rectangle ``(0, 0, 0, 0)``.
        Surface dimensions and clipping are not taken into account.

        Contrast with :meth:`path_extents` which is similar,
        but returns non-zero extents for some paths with no inked area,
        (such as a simple line segment).

        Note that :meth:`fill_extents` must necessarily do more work
        to compute the precise inked areas in light of the fill rule,
        so :meth:`path_extents` may be more desirable for sake of performance
        if the non-inked path extents are desired.

        See :meth:`fill`, :meth:`set_fill_rule` and :meth:`fill_preserve`.

        :return:
            A ``(x1, y1, x2, y2)`` tuple of floats:
            the left, top, right and bottom of the resulting extents,
            respectively.

        """
    extents = ffi.new('double[4]')
    cairo.cairo_fill_extents(self._pointer, extents + 0, extents + 1, extents + 2, extents + 3)
    self._check_status()
    return tuple(extents)