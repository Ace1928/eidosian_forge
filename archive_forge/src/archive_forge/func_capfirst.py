import importlib
import logging
import os
import pathlib
import random
import sys
import threading
import time
import urllib.parse
from collections import deque
from types import ModuleType
from typing import (
import numpy as np
import ray
from ray._private.utils import _get_pyarrow_version
from ray.data._internal.arrow_ops.transform_pyarrow import unify_schemas
from ray.data.context import WARN_PREFIX, DataContext
def capfirst(s: str):
    """Capitalize the first letter of a string

    Args:
        s: String to capitalize

    Returns:
       Capitalized string
    """
    return s[0].upper() + s[1:]