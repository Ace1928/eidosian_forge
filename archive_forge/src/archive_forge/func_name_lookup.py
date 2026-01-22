import ast
import inspect
import re
import sys
import warnings
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union
from .. import language
from .._C.libtriton.triton import ir
from ..language import constexpr, tensor
from ..runtime import JITFunction
from .errors import (CompilationError, CompileTimeAssertionFailure, UnsupportedLanguageConstruct)
def name_lookup(name: str) -> Any:
    absent = absent_marker
    for lookup_function in (local_lookup, self.gscope.get, self.builtin_namespace.get):
        value = lookup_function(name, absent)
        if value is not absent:
            return value
    raise NameError(f'{name} is not defined')