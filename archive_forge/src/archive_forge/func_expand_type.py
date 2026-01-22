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
def expand_type(self, current_info: TypeInfo, api: SemanticAnalyzerPluginInterface) -> Type | None:
    """Based on mypy.plugins.dataclasses.DataclassAttribute.expand_type."""
    if self.type is not None and getattr(self.info, 'self_type', None) is not None:
        expanded_type = expand_type(self.type, {self.info.self_type.id: fill_typevars(current_info)})
        if isinstance(self.type, UnionType) and (not isinstance(expanded_type, UnionType)):
            if not api.final_iteration:
                raise _DeferAnalysis()
        return expanded_type
    return self.type