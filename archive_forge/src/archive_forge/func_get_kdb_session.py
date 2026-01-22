from __future__ import annotations
import os
import abc
import contextlib
import multiprocessing
from pathlib import Path
from .types import AppEnv
from lazyops.libs.proxyobj import proxied
from typing import Optional, Dict, Any, List, Union, Type, Callable, TypeVar, Literal, overload, TYPE_CHECKING
def get_kdb_session(self, name: Optional[str]=None, serializer: Optional[str]='json', **kwargs) -> 'KVDBSession':
    """
        Retrieves or Initializes a KVDB Session
        """
    name = name or self.module_name
    if name not in self._kdb_sessions:
        from kvdb import KVDBClient
        self._kdb_sessions[name] = KVDBClient.get_session(name=name, serializer=serializer, **kwargs)
    return self._kdb_sessions[name]