import inspect
import os
import random
import shutil
import tempfile
import weakref
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import xxhash
from . import config
from .naming import INVALID_WINDOWS_CHARACTERS_IN_PATH
from .utils._dill import dumps
from .utils.deprecation_utils import deprecated
from .utils.logging import get_logger
def format_transform_for_fingerprint(func: Callable, version: Optional[str]=None) -> str:
    """
    Format a transform to the format that will be used to update the fingerprint.
    """
    transform = f'{func.__module__}.{func.__qualname__}'
    if version is not None:
        transform += f'@{version}'
    return transform