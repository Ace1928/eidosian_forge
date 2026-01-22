from __future__ import annotations
import os
import abc
import contextlib
import multiprocessing
from pathlib import Path
from .types import AppEnv
from lazyops.libs.proxyobj import proxied
from typing import Optional, Dict, Any, List, Union, Type, Callable, TypeVar, Literal, overload, TYPE_CHECKING
def get_ctx(self, module_name: str, *args, **kwargs) -> ApplicationContext:
    """
        Retrieves the app context
        """
    if module_name not in self.ctxs:
        return self.init_ctx(module_name, *args, **kwargs)
    return self.ctxs[module_name]