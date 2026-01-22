import importlib.util
import inspect
import sys
from collections import OrderedDict
from contextlib import contextmanager
from typing import Tuple, Union
import numpy as np
from packaging import version
from transformers.utils import is_torch_available
def check_if_transformers_greater(target_version: Union[str, version.Version]) -> bool:
    """
    Checks whether the current install of transformers is greater than or equal to the target version.

    Args:
        target_version (`Union[str, packaging.version.Version]`): version used as the reference for comparison.

    Returns:
        bool: whether the check is True or not.
    """
    import transformers
    if isinstance(target_version, str):
        target_version = version.parse(target_version)
    return version.parse(transformers.__version__) >= target_version