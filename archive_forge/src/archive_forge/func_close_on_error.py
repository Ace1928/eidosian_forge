from __future__ import annotations
import contextlib
import functools
import inspect
import io
import itertools
import math
import os
import re
import sys
import warnings
from collections.abc import (
from enum import Enum
from pathlib import Path
from typing import (
import numpy as np
import pandas as pd
from xarray.namedarray.utils import (  # noqa: F401
@contextlib.contextmanager
def close_on_error(f):
    """Context manager to ensure that a file opened by xarray is closed if an
    exception is raised before the user sees the file object.
    """
    try:
        yield
    except Exception:
        f.close()
        raise