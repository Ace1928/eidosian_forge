from pythran.types.conversion import pytype_to_pretty_type
from collections import defaultdict
from itertools import product
import re
import ply.lex as lex
import ply.yacc as yacc
from pythran.typing import List, Set, Dict, NDArray, Tuple, Pointer, Fun
from pythran.syntax import PythranSyntaxError
from pythran.config import cfg
def p_exports(self, p):
    if len(p) > 1:
        isnative = len(p) == 6
        target = self.exports if len(p) == 6 else self.native_exports
        for key, val in p[len(p) - 3]:
            target[key] += val