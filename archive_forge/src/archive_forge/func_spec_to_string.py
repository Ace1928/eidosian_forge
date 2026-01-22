from pythran.types.conversion import pytype_to_pretty_type
from collections import defaultdict
from itertools import product
import re
import ply.lex as lex
import ply.yacc as yacc
from pythran.typing import List, Set, Dict, NDArray, Tuple, Pointer, Fun
from pythran.syntax import PythranSyntaxError
from pythran.config import cfg
def spec_to_string(function_name, spec):
    arguments_types = [pytype_to_pretty_type(t) for t in spec]
    return '{}({})'.format(function_name, ', '.join(arguments_types))