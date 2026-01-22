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
def torch_only_method(fn):

    def wrapper(*args, **kwargs):
        if not _torch_available:
            raise ImportError('You need to install pytorch to use this method or class, or activate it with environment variables USE_TORCH=1 and USE_TF=0.')
        else:
            return fn(*args, **kwargs)
    return wrapper