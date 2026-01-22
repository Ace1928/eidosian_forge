from __future__ import annotations
import re
import os
import sys
import logging
import typing
import traceback
import warnings
import pprint
import atexit as _atexit
import functools
import threading
from enum import Enum
from loguru import _defaults
from loguru._logger import Core as _Core
from loguru._logger import Logger as _Logger
from typing import Type, Union, Optional, Any, List, Dict, Tuple, Callable, Set, TYPE_CHECKING
def mute_warning(self, action: str='ignore', category: Type[Warning]=Warning, module: str=None, **kwargs):
    """
        Helper to mute a warning from another module.
        """
    if module:
        warnings.filterwarnings(action, category=category, module=module, **kwargs)
    else:
        warnings.filterwarnings(action, category=category, **kwargs)