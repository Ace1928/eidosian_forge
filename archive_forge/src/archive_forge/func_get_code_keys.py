import copy
import dataclasses
import dis
import itertools
import sys
import types
from typing import Any, Callable, cast, Dict, Iterator, List, Optional, Tuple
from .bytecode_analysis import (
def get_code_keys() -> List[str]:
    keys = ['co_argcount']
    keys.append('co_posonlyargcount')
    keys.extend(['co_kwonlyargcount', 'co_nlocals', 'co_stacksize', 'co_flags', 'co_code', 'co_consts', 'co_names', 'co_varnames', 'co_filename', 'co_name'])
    if sys.version_info >= (3, 11):
        keys.append('co_qualname')
    keys.append('co_firstlineno')
    if sys.version_info >= (3, 10):
        keys.append('co_linetable')
    else:
        keys.append('co_lnotab')
    if sys.version_info >= (3, 11):
        keys.append('co_exceptiontable')
    keys.extend(['co_freevars', 'co_cellvars'])
    return keys