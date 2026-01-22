from pythran.types.conversion import pytype_to_pretty_type
from collections import defaultdict
from itertools import product
import re
import ply.lex as lex
import ply.yacc as yacc
from pythran.typing import List, Set, Dict, NDArray, Tuple, Pointer, Fun
from pythran.syntax import PythranSyntaxError
from pythran.config import cfg
def p_param_types(self, p):
    if len(p) == 2 or (len(p) == 3 and p[2] == ','):
        p[0] = tuple(((t,) for t in p[1]))
    elif len(p) == 3 and p[2] == '?':
        p[0] = tuple(((t,) for t in p[1])) + ((),)
    elif len(p) == 4:
        p[0] = tuple(((t,) + ts for t in p[1] for ts in p[3]))
    else:
        p[0] = tuple(((t,) + ts for t in p[1] for ts in p[4])) + ((),)