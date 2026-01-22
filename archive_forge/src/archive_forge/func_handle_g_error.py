import sys
from array import array
from functools import partial
from io import BytesIO
from . import Context, ImageSurface, constants, dlopen
from .ffi import ffi_pixbuf as ffi
def handle_g_error(error, return_value):
    """Convert a ``GError**`` to a Python :exception:`ImageLoadingError`,
    and raise it.

    """
    error = error[0]
    assert bool(return_value) == (error == ffi.NULL)
    if error != ffi.NULL:
        if error.message != ffi.NULL:
            message = 'Pixbuf error: ' + ffi.string(error.message).decode('utf8', 'replace')
        else:
            message = 'Pixbuf error'
        glib.g_error_free(error)
        raise ImageLoadingError(message)