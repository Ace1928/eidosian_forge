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
def is_ipex_available():

    def get_major_and_minor_from_version(full_version):
        return str(version.parse(full_version).major) + '.' + str(version.parse(full_version).minor)
    if not is_torch_available() or not _ipex_available:
        return False
    torch_major_and_minor = get_major_and_minor_from_version(_torch_version)
    ipex_major_and_minor = get_major_and_minor_from_version(_ipex_version)
    if torch_major_and_minor != ipex_major_and_minor:
        logger.warning(f'Intel Extension for PyTorch {ipex_major_and_minor} needs to work with PyTorch {ipex_major_and_minor}.*, but PyTorch {_torch_version} is found. Please switch to the matching version and run again.')
        return False
    return True