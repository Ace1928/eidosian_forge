from __future__ import annotations
import os
import abc
import contextlib
import multiprocessing
from pathlib import Path
from .types import AppEnv
from lazyops.libs.proxyobj import proxied
from typing import Optional, Dict, Any, List, Union, Type, Callable, TypeVar, Literal, overload, TYPE_CHECKING
def register_settings(self, settings: 'AppSettingsT') -> None:
    """
        Registers the settings
        """
    from .lazy import register_module_settings
    register_module_settings(self.module_name, settings)
    self._settings = settings