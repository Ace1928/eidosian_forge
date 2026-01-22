import bisect
import sys
import logging
import os
import os.path
import ply.lex as lex
import ply.yacc as yacc
from inspect import getfile, currentframe
from pyomo.common.fileutils import this_file
from pyomo.core.base.util import flatten_tuple
@lex.TOKEN(_re_number + '(?=[\\s()\\[\\]{}:;,])')
def t_NUM_VAL(t):
    _num = float(t.value)
    if '.' in t.value:
        t.value = _num
    else:
        _int = int(_num)
        t.value = _int if _num == _int else _num
    return t