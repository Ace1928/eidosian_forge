import importlib.metadata
import platform
import sys
import warnings
from typing import Any, Dict
from .. import __version__, constants
def get_tensorboard_version() -> str:
    return _get_version('tensorboard')