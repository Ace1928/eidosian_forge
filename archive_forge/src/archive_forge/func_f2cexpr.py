from . import __version__
import copy
import re
import os
from .crackfortran import markoutercomma
from . import cb_rules
from ._isocbind import iso_c_binding_map, isoc_c2pycode_map, iso_c2py_map
from .auxfuncs import *
def f2cexpr(expr):
    """Rewrite Fortran expression as f2py supported C expression.

    Due to the lack of a proper expression parser in f2py, this
    function uses a heuristic approach that assumes that Fortran
    arithmetic expressions are valid C arithmetic expressions when
    mapping Fortran function calls to the corresponding C function/CPP
    macros calls.

    """
    expr = re.sub('\\blen\\b', 'f2py_slen', expr)
    return expr