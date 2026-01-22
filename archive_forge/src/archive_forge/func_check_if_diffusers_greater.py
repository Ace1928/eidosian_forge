import importlib.util
import inspect
import sys
from collections import OrderedDict
from contextlib import contextmanager
from typing import Tuple, Union
import numpy as np
from packaging import version
from transformers.utils import is_torch_available
def check_if_diffusers_greater(target_version: str) -> bool:
    """
    Checks whether the current install of diffusers is greater than or equal to the target version.

    Args:
        target_version (str): version used as the reference for comparison.

    Returns:
        bool: whether the check is True or not.
    """
    if not _diffusers_available:
        return False
    return version.parse(_diffusers_version) >= version.parse(target_version)