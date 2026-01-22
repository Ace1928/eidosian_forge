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
def is_accelerate_available(min_version: str=ACCELERATE_MIN_VERSION):
    if min_version is not None:
        return _accelerate_available and version.parse(_accelerate_version) >= version.parse(min_version)
    return _accelerate_available