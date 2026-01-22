from __future__ import annotations
import os
import abc
import contextlib
import multiprocessing
from pathlib import Path
from .types import AppEnv
from lazyops.libs.proxyobj import proxied
from typing import Optional, Dict, Any, List, Union, Type, Callable, TypeVar, Literal, overload, TYPE_CHECKING
def get_component_schema(self, name: str, *parts: str, kind: Optional[str]=None, include_kind: Optional[bool]=None) -> Type['ComponentSchemaT']:
    """
        Gets a component schema
        """
    include_kind = include_kind if include_kind is not None else self.include_kind_in_component_name
    if include_kind:
        schema_name = f'{self.module_name}.{kind}' if kind else self.module_name
    else:
        schema_name = self.module_name
    if parts:
        parts = '.'.join(parts)
        schema_name = f'{schema_name}.{parts}'
    schema_name = f'{schema_name}.{name}'
    if schema_name not in self._component_schema_registry:
        raise ValueError(f'Invalid component schema: {schema_name}')
    schema = self._component_schema_registry[schema_name]
    if isinstance(schema, str):
        from lazyops.utils.lazy import lazy_import
        schema = lazy_import(schema)
    return schema