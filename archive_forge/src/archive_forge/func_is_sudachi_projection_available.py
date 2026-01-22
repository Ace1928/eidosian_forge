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
def is_sudachi_projection_available():
    if not is_sudachi_available():
        return False
    return version.parse(_sudachipy_version) >= version.parse('0.6.8')