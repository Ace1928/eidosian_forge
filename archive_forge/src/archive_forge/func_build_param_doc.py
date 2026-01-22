import re
import atexit
import ctypes
import os
import sys
import inspect
import platform
import numpy as _np
from . import libinfo
def build_param_doc(arg_names, arg_types, arg_descs, remove_dup=True):
    """Build argument docs in python style.

    arg_names : list of str
        Argument names.

    arg_types : list of str
        Argument type information.

    arg_descs : list of str
        Argument description information.

    remove_dup : boolean, optional
        Whether remove duplication or not.

    Returns
    -------
    docstr : str
        Python docstring of parameter sections.
    """
    param_keys = set()
    param_str = []
    for key, type_info, desc in zip(arg_names, arg_types, arg_descs):
        if key in param_keys and remove_dup:
            continue
        if key == 'num_args':
            continue
        param_keys.add(key)
        ret = '%s : %s' % (key, type_info)
        if len(desc) != 0:
            ret += '\n    ' + desc
        param_str.append(ret)
    doc_str = 'Parameters\n' + '----------\n' + '%s\n'
    doc_str = doc_str % '\n'.join(param_str)
    return doc_str