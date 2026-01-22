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
def fz_new_image_from_file(path):
    """
    Class-aware wrapper for `::fz_new_image_from_file()`.
    	Create a new image from the contents
    	of a file, inferring its type from the format of the
    	data.
    """
    return _mupdf.fz_new_image_from_file(path)