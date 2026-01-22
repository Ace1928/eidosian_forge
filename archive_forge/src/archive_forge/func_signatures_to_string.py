from pythran.types.conversion import pytype_to_pretty_type
from collections import defaultdict
from itertools import product
import re
import ply.lex as lex
import ply.yacc as yacc
from pythran.typing import List, Set, Dict, NDArray, Tuple, Pointer, Fun
from pythran.syntax import PythranSyntaxError
from pythran.config import cfg
def signatures_to_string(func_name, signatures):
    sigdocs = [spec_to_string(func_name, sig) for sig in signatures if not any((istransposed(t) for t in sig))]
    if not sigdocs:
        sigdocs = [spec_to_string(func_name, sig) for sig in signatures]
    return ''.join(('\n    - ' + sigdoc for sigdoc in sigdocs))