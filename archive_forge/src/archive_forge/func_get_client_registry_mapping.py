from __future__ import annotations
import os
import abc
import contextlib
import multiprocessing
from pathlib import Path
from .types import AppEnv
from lazyops.libs.proxyobj import proxied
from typing import Optional, Dict, Any, List, Union, Type, Callable, TypeVar, Literal, overload, TYPE_CHECKING
def get_client_registry_mapping(self) -> Dict[str, str]:
    """
        Retrieves the client registry mapping
        """
    return {**self._global_clients, **self._local_clients}