from __future__ import annotations
import os
import abc
import contextlib
import multiprocessing
from pathlib import Path
from .types import AppEnv
from lazyops.libs.proxyobj import proxied
from typing import Optional, Dict, Any, List, Union, Type, Callable, TypeVar, Literal, overload, TYPE_CHECKING
@property
def global_ctx(self):
    """
        Returns the global context
        """
    if self._global_ctx is None:
        from ..state import GlobalContext
        self._global_ctx = GlobalContext
    return self._global_ctx