import sys
import string
import fileinput
import re
import os
import copy
import platform
import codecs
from pathlib import Path
from . import __version__
from .auxfuncs import *
from . import symbolic
def param_eval(v, g_params, params, dimspec=None):
    """
    Creates a dictionary of indices and values for each parameter in a
    parameter array to be evaluated later.

    WARNING: It is not possible to initialize multidimensional array
    parameters e.g. dimension(-3:1, 4, 3:5) at this point. This is because in
    Fortran initialization through array constructor requires the RESHAPE
    intrinsic function. Since the right-hand side of the parameter declaration
    is not executed in f2py, but rather at the compiled c/fortran extension,
    later, it is not possible to execute a reshape of a parameter array.
    One issue remains: if the user wants to access the array parameter from
    python, we should either
    1) allow them to access the parameter array using python standard indexing
       (which is often incompatible with the original fortran indexing)
    2) allow the parameter array to be accessed in python as a dictionary with
       fortran indices as keys
    We are choosing 2 for now.
    """
    if dimspec is None:
        try:
            p = eval(v, g_params, params)
        except Exception as msg:
            p = v
            outmess(f'param_eval: got "{msg}" on {v!r}\n')
        return p
    if len(dimspec) < 2 or dimspec[::len(dimspec) - 1] != '()':
        raise ValueError(f"param_eval: dimension {dimspec} can't be parsed")
    dimrange = dimspec[1:-1].split(',')
    if len(dimrange) == 1:
        dimrange = dimrange[0].split(':')
        if len(dimrange) == 1:
            bound = param_parse(dimrange[0], params)
            dimrange = range(1, int(bound) + 1)
        else:
            lbound = param_parse(dimrange[0], params)
            ubound = param_parse(dimrange[1], params)
            dimrange = range(int(lbound), int(ubound) + 1)
    else:
        raise ValueError(f'param_eval: multidimensional array parameters {{dimspec}} not supported')
    v = (v[2:-2] if v.startswith('(/') else v).split(',')
    v_eval = []
    for item in v:
        try:
            item = eval(item, g_params, params)
        except Exception as msg:
            outmess(f'param_eval: got "{msg}" on {item!r}\n')
        v_eval.append(item)
    p = dict(zip(dimrange, v_eval))
    return p