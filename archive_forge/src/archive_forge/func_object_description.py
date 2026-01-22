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
def object_description(object: Any) -> str:
    """A repr() implementation that returns text safe to use in reST context."""
    if isinstance(object, dict):
        try:
            sorted_keys = sorted(object)
        except Exception:
            pass
        else:
            items = ('%s: %s' % (object_description(key), object_description(object[key])) for key in sorted_keys)
            return '{%s}' % ', '.join(items)
    elif isinstance(object, set):
        try:
            sorted_values = sorted(object)
        except TypeError:
            pass
        else:
            return '{%s}' % ', '.join((object_description(x) for x in sorted_values))
    elif isinstance(object, frozenset):
        try:
            sorted_values = sorted(object)
        except TypeError:
            pass
        else:
            return 'frozenset({%s})' % ', '.join((object_description(x) for x in sorted_values))
    elif isinstance(object, enum.Enum):
        return '%s.%s' % (object.__class__.__name__, object.name)
    try:
        s = repr(object)
    except Exception as exc:
        raise ValueError from exc
    s = memory_address_re.sub('', s)
    return s.replace('\n', ' ')