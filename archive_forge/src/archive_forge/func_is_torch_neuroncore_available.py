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
@lru_cache()
def is_torch_neuroncore_available(check_device=True):
    if importlib.util.find_spec('torch_neuronx') is not None:
        return is_torch_tpu_available(check_device)
    return False