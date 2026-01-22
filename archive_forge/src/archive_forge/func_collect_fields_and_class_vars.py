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
def collect_fields_and_class_vars(self, model_config: ModelConfigData, is_root_model: bool) -> tuple[list[PydanticModelField] | None, list[PydanticModelClassVar] | None]:
    """Collects the fields for the model, accounting for parent classes."""
    cls = self._cls
    found_fields: dict[str, PydanticModelField] = {}
    found_class_vars: dict[str, PydanticModelClassVar] = {}
    for info in reversed(cls.info.mro[1:-1]):
        if METADATA_KEY not in info.metadata:
            continue
        self._api.add_plugin_dependency(make_wildcard_trigger(info.fullname))
        for name, data in info.metadata[METADATA_KEY]['fields'].items():
            field = PydanticModelField.deserialize(info, data, self._api)
            with state.strict_optional_set(self._api.options.strict_optional):
                field.expand_typevar_from_subtype(cls.info)
            found_fields[name] = field
            sym_node = cls.info.names.get(name)
            if sym_node and sym_node.node and (not isinstance(sym_node.node, Var)):
                self._api.fail('BaseModel field may only be overridden by another field', sym_node.node)
        for name, data in info.metadata[METADATA_KEY]['class_vars'].items():
            found_class_vars[name] = PydanticModelClassVar.deserialize(data)
    current_field_names: set[str] = set()
    current_class_vars_names: set[str] = set()
    for stmt in self._get_assignment_statements_from_block(cls.defs):
        maybe_field = self.collect_field_or_class_var_from_stmt(stmt, model_config, found_class_vars)
        if isinstance(maybe_field, PydanticModelField):
            lhs = stmt.lvalues[0]
            if is_root_model and lhs.name != 'root':
                error_extra_fields_on_root_model(self._api, stmt)
            else:
                current_field_names.add(lhs.name)
                found_fields[lhs.name] = maybe_field
        elif isinstance(maybe_field, PydanticModelClassVar):
            lhs = stmt.lvalues[0]
            current_class_vars_names.add(lhs.name)
            found_class_vars[lhs.name] = maybe_field
    return (list(found_fields.values()), list(found_class_vars.values()))