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
def collect_config(self) -> 'ModelConfigData':
    """
        Collects the values of the config attributes that are used by the plugin, accounting for parent classes.
        """
    ctx = self._ctx
    cls = ctx.cls
    config = ModelConfigData()
    for stmt in cls.defs.body:
        if not isinstance(stmt, ClassDef):
            continue
        if stmt.name == 'Config':
            for substmt in stmt.defs.body:
                if not isinstance(substmt, AssignmentStmt):
                    continue
                config.update(self.get_config_update(substmt))
            if config.has_alias_generator and (not config.allow_population_by_field_name) and self.plugin_config.warn_required_dynamic_aliases:
                error_required_dynamic_aliases(ctx.api, stmt)
    for info in cls.info.mro[1:]:
        if METADATA_KEY not in info.metadata:
            continue
        ctx.api.add_plugin_dependency(make_wildcard_trigger(get_fullname(info)))
        for name, value in info.metadata[METADATA_KEY]['config'].items():
            config.setdefault(name, value)
    return config