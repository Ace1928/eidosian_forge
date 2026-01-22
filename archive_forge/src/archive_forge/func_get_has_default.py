from __future__ import annotations
import sys
from configparser import ConfigParser
from typing import Any, Callable, Iterator
from mypy.errorcodes import ErrorCode
from mypy.expandtype import expand_type, expand_type_by_instance
from mypy.nodes import (
from mypy.options import Options
from mypy.plugin import (
from mypy.plugins import dataclasses
from mypy.plugins.common import (
from mypy.semanal import set_callable_name
from mypy.server.trigger import make_wildcard_trigger
from mypy.state import state
from mypy.typeops import map_type_from_supertype
from mypy.types import (
from mypy.typevars import fill_typevars
from mypy.util import get_unique_redefinition_name
from mypy.version import __version__ as mypy_version
from pydantic._internal import _fields
from pydantic.version import parse_mypy_version
@staticmethod
def get_has_default(stmt: AssignmentStmt) -> bool:
    """Returns a boolean indicating whether the field defined in `stmt` is a required field."""
    expr = stmt.rvalue
    if isinstance(expr, TempNode):
        return False
    if isinstance(expr, CallExpr) and isinstance(expr.callee, RefExpr) and (expr.callee.fullname == FIELD_FULLNAME):
        for arg, name in zip(expr.args, expr.arg_names):
            if name is None or name == 'default':
                return arg.__class__ is not EllipsisExpr
            if name == 'default_factory':
                return not (isinstance(arg, NameExpr) and arg.fullname == 'builtins.None')
        return False
    return not isinstance(expr, EllipsisExpr)