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
def validate_fingerprint(fingerprint: str, max_length=64):
    """
    Make sure the fingerprint is a non-empty string that is not longer that max_length=64 by default,
    so that the fingerprint can be used to name cache files without issues.
    """
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ValueError(f"Invalid fingerprint '{fingerprint}': it should be a non-empty string.")
    for invalid_char in INVALID_WINDOWS_CHARACTERS_IN_PATH:
        if invalid_char in fingerprint:
            raise ValueError(f"Invalid fingerprint. Bad characters from black list '{INVALID_WINDOWS_CHARACTERS_IN_PATH}' found in '{fingerprint}'. They could create issues when creating cache files.")
    if len(fingerprint) > max_length:
        raise ValueError(f"Invalid fingerprint. Maximum lenth is {max_length} but '{fingerprint}' has length {len(fingerprint)}.It could create issues when creating cache files.")