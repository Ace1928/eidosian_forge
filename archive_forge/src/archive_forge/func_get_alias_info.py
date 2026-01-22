import sys
from configparser import ConfigParser
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type as TypingType, Union
from mypy.errorcodes import ErrorCode
from mypy.nodes import (
from mypy.options import Options
from mypy.plugin import (
from mypy.plugins import dataclasses
from mypy.semanal import set_callable_name  # type: ignore
from mypy.server.trigger import make_wildcard_trigger
from mypy.types import (
from mypy.typevars import fill_typevars
from mypy.util import get_unique_redefinition_name
from mypy.version import __version__ as mypy_version
from pydantic.utils import is_valid_field
@staticmethod
def get_alias_info(stmt: AssignmentStmt) -> Tuple[Optional[str], bool]:
    """
        Returns a pair (alias, has_dynamic_alias), extracted from the declaration of the field defined in `stmt`.

        `has_dynamic_alias` is True if and only if an alias is provided, but not as a string literal.
        If `has_dynamic_alias` is True, `alias` will be None.
        """
    expr = stmt.rvalue
    if isinstance(expr, TempNode):
        return (None, False)
    if not (isinstance(expr, CallExpr) and isinstance(expr.callee, RefExpr) and (expr.callee.fullname == FIELD_FULLNAME)):
        return (None, False)
    for i, arg_name in enumerate(expr.arg_names):
        if arg_name != 'alias':
            continue
        arg = expr.args[i]
        if isinstance(arg, StrExpr):
            return (arg.value, False)
        else:
            return (None, True)
    return (None, False)