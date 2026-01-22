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
def is_bitsandbytes_available():
    if not is_torch_available():
        return False
    import torch
    return _bitsandbytes_available and torch.cuda.is_available()