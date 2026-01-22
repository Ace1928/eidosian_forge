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
def ll_pdf_walk_tree(tree, kid_name, arrive, leave, arg):
    """
    Wrapper for out-params of pdf_walk_tree().
    Returns: ::pdf_obj *names, ::pdf_obj *values
    """
    outparams = ll_pdf_walk_tree_outparams()
    ret = ll_pdf_walk_tree_outparams_fn(tree, kid_name, arrive, leave, arg, outparams)
    return (outparams.names, outparams.values)