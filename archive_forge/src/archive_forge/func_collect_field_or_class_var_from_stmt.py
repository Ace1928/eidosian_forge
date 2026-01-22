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
def collect_field_or_class_var_from_stmt(self, stmt: AssignmentStmt, model_config: ModelConfigData, class_vars: dict[str, PydanticModelClassVar]) -> PydanticModelField | PydanticModelClassVar | None:
    """Get pydantic model field from statement.

        Args:
            stmt: The statement.
            model_config: Configuration settings for the model.
            class_vars: ClassVars already known to be defined on the model.

        Returns:
            A pydantic model field if it could find the field in statement. Otherwise, `None`.
        """
    cls = self._cls
    lhs = stmt.lvalues[0]
    if not isinstance(lhs, NameExpr) or not _fields.is_valid_field_name(lhs.name) or lhs.name == 'model_config':
        return None
    if not stmt.new_syntax:
        if isinstance(stmt.rvalue, CallExpr) and isinstance(stmt.rvalue.callee, CallExpr) and isinstance(stmt.rvalue.callee.callee, NameExpr) and (stmt.rvalue.callee.callee.fullname in DECORATOR_FULLNAMES):
            return None
        if lhs.name in class_vars:
            return None
        error_untyped_fields(self._api, stmt)
        return None
    lhs = stmt.lvalues[0]
    if not isinstance(lhs, NameExpr):
        return None
    if not _fields.is_valid_field_name(lhs.name) or lhs.name == 'model_config':
        return None
    sym = cls.info.names.get(lhs.name)
    if sym is None:
        return None
    node = sym.node
    if isinstance(node, PlaceholderNode):
        return None
    if isinstance(node, TypeAlias):
        self._api.fail('Type aliases inside BaseModel definitions are not supported at runtime', node)
        return None
    if not isinstance(node, Var):
        return None
    if node.is_classvar:
        return PydanticModelClassVar(lhs.name)
    node_type = get_proper_type(node.type)
    if isinstance(node_type, Instance) and node_type.type.fullname == 'dataclasses.InitVar':
        self._api.fail('InitVar is not supported in BaseModel', node)
    has_default = self.get_has_default(stmt)
    if sym.type is None and node.is_final and node.is_inferred:
        typ = self._api.analyze_simple_literal_type(stmt.rvalue, is_final=True)
        if typ:
            node.type = typ
        else:
            self._api.fail('Need type argument for Final[...] with non-literal default in BaseModel', stmt)
            node.type = AnyType(TypeOfAny.from_error)
    alias, has_dynamic_alias = self.get_alias_info(stmt)
    if has_dynamic_alias and (not model_config.populate_by_name) and self.plugin_config.warn_required_dynamic_aliases:
        error_required_dynamic_aliases(self._api, stmt)
    init_type = self._infer_dataclass_attr_init_type(sym, lhs.name, stmt)
    return PydanticModelField(name=lhs.name, has_dynamic_alias=has_dynamic_alias, has_default=has_default, alias=alias, line=stmt.line, column=stmt.column, type=init_type, info=cls.info)