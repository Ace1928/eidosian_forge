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
def fz_new_draw_device_with_options_outparams_fn(options, mediabox):
    """
    Class-aware helper for out-params of fz_new_draw_device_with_options() [fz_new_draw_device_with_options()].
    """
    ret, pixmap = ll_fz_new_draw_device_with_options(options.internal(), mediabox.internal())
    return (FzDevice(ret), FzPixmap(pixmap))