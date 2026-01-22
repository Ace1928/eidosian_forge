from pythran.types.conversion import pytype_to_pretty_type
from collections import defaultdict
from itertools import product
import re
import ply.lex as lex
import ply.yacc as yacc
from pythran.typing import List, Set, Dict, NDArray, Tuple, Pointer, Fun
from pythran.syntax import PythranSyntaxError
from pythran.config import cfg
def istransposable(t):
    if not isinstance(t, NDArray):
        return False
    if len(t.__args__) - 1 != 2:
        return False
    return all((s.step == 1 for s in t.__args__[1:]))