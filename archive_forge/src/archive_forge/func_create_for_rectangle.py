import ctypes
import io
import operator
import os
import sys
import weakref
from functools import reduce
from pathlib import Path
from tempfile import NamedTemporaryFile
from . import _check_status, _keepref, cairo, constants, ffi
from .fonts import FontOptions, _encode_string
def create_for_rectangle(self, x, y, width, height):
    """
        Create a new surface that is a rectangle within this surface.
        All operations drawn to this surface are then clipped and translated
        onto the target surface.
        Nothing drawn via this sub-surface outside of its bounds
        is drawn onto the target surface,
        making this a useful method for passing constrained child surfaces
        to library routines that draw directly onto the parent surface,
        i.e. with no further backend allocations,
        double buffering or copies.

        .. note::

            As of cairo 1.12,
            the semantics of subsurfaces have not been finalized yet
            unless the rectangle is in full device units,
            is contained within the extents of the target surface,
            and the target or subsurface's device transforms are not changed.

        :param x:
            The x-origin of the sub-surface
            from the top-left of the target surface (in device-space units)
        :param y:
            The y-origin of the sub-surface
            from the top-left of the target surface (in device-space units)
        :param width:
            Width of the sub-surface (in device-space units)
        :param height:
            Height of the sub-surface (in device-space units)
        :type x: float
        :type y: float
        :type width: float
        :type height: float
        :returns:
            A new :class:`Surface` object.

        *New in cairo 1.10.*

        """
    return Surface._from_pointer(cairo.cairo_surface_create_for_rectangle(self._pointer, x, y, width, height), incref=False)