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
def visit_Assert(self, node) -> Any:
    if not self.debug:
        return
    test = self.visit(node.test)
    msg = self.visit(node.msg)
    return language.core.device_assert(test, msg, _builder=self.builder)