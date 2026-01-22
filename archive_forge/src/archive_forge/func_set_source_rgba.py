from . import _check_status, _keepref, cairo, constants, ffi
from .fonts import FontFace, FontOptions, ScaledFont, _encode_string
from .matrix import Matrix
from .patterns import Pattern
from .surfaces import Surface
def set_source_rgba(self, red, green, blue, alpha=1):
    """Sets the source pattern within this context to a solid color.
        This color will then be used for any subsequent drawing operation
        until a new source pattern is set.

        The color and alpha components are
        floating point numbers  in the range 0 to 1.
        If the values passed in are outside that range, they will be clamped.

        The default source pattern is opaque black,
        (that is, it is equivalent to ``context.set_source_rgba(0, 0, 0)``).

        :param red: Red component of the color.
        :param green: Green component of the color.
        :param blue: Blue component of the color.
        :param alpha:
            Alpha component of the color.
            1 (the default) is opaque, 0 fully transparent.
        :type red: float
        :type green: float
        :type blue: float
        :type alpha: float

        """
    cairo.cairo_set_source_rgba(self._pointer, red, green, blue, alpha)
    self._check_status()