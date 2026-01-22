from sys import version_info as _swig_python_version_info
import weakref
import inspect
import os
import re
import sys
import traceback
import inspect
import io
import os
import sys
import traceback
import types
def fz_compressed_image_buffer(self):
    """
        Class-aware wrapper for `::fz_compressed_image_buffer()`.
        	Retrieve the underlying compressed data for an image.

        	Returns a pointer to the underlying data buffer for an image,
        	or NULL if this image is not based upon a compressed data
        	buffer.

        	This is not a reference counted structure, so no reference is
        	returned. Lifespan is limited to that of the image itself.
        """
    return _mupdf.FzImage_fz_compressed_image_buffer(self)