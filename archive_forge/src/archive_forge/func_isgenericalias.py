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
def isgenericalias(obj: Any) -> bool:
    """Check if the object is GenericAlias."""
    if hasattr(typing, '_GenericAlias') and isinstance(obj, typing._GenericAlias):
        return True
    elif hasattr(types, 'GenericAlias') and isinstance(obj, types.GenericAlias):
        return True
    elif hasattr(typing, '_SpecialGenericAlias') and isinstance(obj, typing._SpecialGenericAlias):
        return True
    else:
        return False