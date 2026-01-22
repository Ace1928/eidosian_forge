import sys
from array import array
from functools import partial
from io import BytesIO
from . import Context, ImageSurface, constants, dlopen
from .ffi import ffi_pixbuf as ffi
Convert from PixBuf to ImageSurface, by going through the PNG format.

    This method is 10~30x slower than GDK but always works.

    