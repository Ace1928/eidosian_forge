from . import _check_status, _keepref, cairo, constants, ffi
from .fonts import FontFace, FontOptions, ScaledFont, _encode_string
from .matrix import Matrix
from .patterns import Pattern
from .surfaces import Surface
def reset_clip(self):
    """Reset the current clip region to its original, unrestricted state.
        That is, set the clip region to an infinitely large shape
        containing the target surface.
        Equivalently, if infinity is too hard to grasp,
        one can imagine the clip region being reset
        to the exact bounds of the target surface.

        Note that code meant to be reusable
        should not call :meth:`reset_clip`
        as it will cause results unexpected by higher-level code
        which calls :meth:`clip`.
        Consider using :meth:`cairo` and :meth:`restore` around :meth:`clip`
        as a more robust means of temporarily restricting the clip region.

        """
    cairo.cairo_reset_clip(self._pointer)
    self._check_status()