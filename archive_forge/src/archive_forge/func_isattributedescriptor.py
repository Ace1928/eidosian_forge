import builtins
import contextlib
import enum
import inspect
import re
import sys
import types
import typing
from functools import partial, partialmethod
from importlib import import_module
from inspect import Parameter, isclass, ismethod, ismethoddescriptor, ismodule  # NOQA
from io import StringIO
from types import MethodType, ModuleType
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Type, cast
from sphinx.pycode.ast import ast  # for py36-37
from sphinx.pycode.ast import unparse as ast_unparse
from sphinx.util import logging
from sphinx.util.typing import ForwardRef
from sphinx.util.typing import stringify as stringify_annotation
def isattributedescriptor(obj: Any) -> bool:
    """Check if the object is an attribute like descriptor."""
    if inspect.isdatadescriptor(obj):
        return True
    elif isdescriptor(obj):
        unwrapped = unwrap(obj)
        if isfunction(unwrapped) or isbuiltin(unwrapped) or inspect.ismethod(unwrapped):
            return False
        elif is_cython_function_or_method(unwrapped):
            return False
        elif inspect.isclass(unwrapped):
            return False
        elif isinstance(unwrapped, (ClassMethodDescriptorType, MethodDescriptorType, WrapperDescriptorType)):
            return False
        elif type(unwrapped).__name__ == 'instancemethod':
            return False
        else:
            return True
    else:
        return False