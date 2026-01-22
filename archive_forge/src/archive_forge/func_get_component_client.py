from __future__ import annotations
import os
import abc
import contextlib
import multiprocessing
from pathlib import Path
from .types import AppEnv
from lazyops.libs.proxyobj import proxied
from typing import Optional, Dict, Any, List, Union, Type, Callable, TypeVar, Literal, overload, TYPE_CHECKING
def get_component_client(self, name: str, *parts: str, kind: Optional[str]=None, include_kind: Optional[bool]=None) -> 'ClientTypes':
    """
        Gets a component client
        """
    from lazyops.libs.fastapi_utils.state.registry import get_client
    include_kind = include_kind if include_kind is not None else self.include_kind_in_component_name
    if include_kind:
        client_name = f'{self.module_name}.{kind}' if kind else self.module_name
    else:
        client_name = self.module_name
    if parts:
        parts = '.'.join(parts)
        client_name = f'{client_name}.{parts}'
    client_name = f'{client_name}.{name}'
    return get_client(client_name)