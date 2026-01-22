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
def local_lookup(name: str, absent):
    value = self.lscope.get(name, absent)
    if value is not absent and name not in self.local_defs:
        self.global_uses[name] = value
    return value