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
def ll_fz_get_pixmap_from_image(image, subarea, ctm):
    """
    Wrapper for out-params of fz_get_pixmap_from_image().
    Returns: fz_pixmap *, int w, int h
    """
    outparams = ll_fz_get_pixmap_from_image_outparams()
    ret = ll_fz_get_pixmap_from_image_outparams_fn(image, subarea, ctm, outparams)
    return (ret, outparams.w, outparams.h)