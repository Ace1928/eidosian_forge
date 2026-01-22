from __future__ import annotations
import os
import abc
import atexit
import pathlib
import filelock
import contextlib
from lazyops.types import BaseModel, Field
from lazyops.utils.logs import logger
from lazyops.utils.serialization import Json
from typing import Optional, Dict, Any, Set, List, Union, Generator, TYPE_CHECKING
def bind_settings(self, settings: 'BaseSettings'):
    """
        Binds the settings to this state
        """
    self.ctx['settings'] = settings
    self.ctx['process_id'] = os.getpid()
    if hasattr(settings, 'app_module_name'):
        self.app_module_name = settings.app_module_name
    else:
        self.app_module_name = settings.__class__.__module__.split('.')[0]
    if hasattr(settings, 'data_path'):
        self.data_path = settings.data_path
    else:
        from lazyops.utils.assets import get_module_path
        module_path = get_module_path(self.app_module_name)
        self.data_path = module_path.joinpath('.data')
    self.configure_stx()