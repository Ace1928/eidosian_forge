from . import _check_status, _keepref, cairo, constants, ffi
from .fonts import FontFace, FontOptions, ScaledFont, _encode_string
from .matrix import Matrix
from .patterns import Pattern
from .surfaces import Surface
def mask_surface(self, surface, surface_x=0, surface_y=0):
    """A drawing operator that paints the current source
        using the alpha channel of ``surface`` as a mask.
        (Opaque areas of ``surface`` are painted with the source,
        transparent areas are not painted.)

        :param pattern: A :class:`Surface` object.
        :param surface_x: X coordinate at which to place the origin of surface.
        :param surface_y: Y coordinate at which to place the origin of surface.
        :type surface_x: float
        :type surface_y: float

        """
    cairo.cairo_mask_surface(self._pointer, surface._pointer, surface_x, surface_y)
    self._check_status()