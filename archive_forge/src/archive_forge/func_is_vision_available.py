import importlib.metadata
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import warnings
from collections import OrderedDict
from functools import lru_cache
from itertools import chain
from types import ModuleType
from typing import Any, Tuple, Union
from packaging import version
from . import logging
def is_vision_available():
    _pil_available = importlib.util.find_spec('PIL') is not None
    if _pil_available:
        try:
            package_version = importlib.metadata.version('Pillow')
        except importlib.metadata.PackageNotFoundError:
            try:
                package_version = importlib.metadata.version('Pillow-SIMD')
            except importlib.metadata.PackageNotFoundError:
                return False
        logger.debug(f'Detected PIL version {package_version}')
    return _pil_available