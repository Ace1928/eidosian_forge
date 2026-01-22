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
def ll_fz_dom_get_attribute(elt, i):
    """
    Wrapper for out-params of fz_dom_get_attribute().
    Returns: const char *, const char *att
    """
    outparams = ll_fz_dom_get_attribute_outparams()
    ret = ll_fz_dom_get_attribute_outparams_fn(elt, i, outparams)
    return (ret, outparams.att)