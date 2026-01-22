from . import _check_status, _keepref, cairo, constants, ffi
from .fonts import FontFace, FontOptions, ScaledFont, _encode_string
from .matrix import Matrix
from .patterns import Pattern
from .surfaces import Surface
def new_sub_path(self):
    """Begin a new sub-path.
        Note that the existing path is not affected.
        After this call there will be no current point.

        In many cases, this call is not needed
        since new sub-paths are frequently started with :meth:`move_to`.

        A call to :meth:`new_sub_path` is particularly useful
        when beginning a new sub-path with one of the :meth:`arc` calls.
        This makes things easier as it is no longer necessary
        to manually compute the arc's initial coordinates
        for a call to :meth:`move_to`.

        """
    cairo.cairo_new_sub_path(self._pointer)
    self._check_status()