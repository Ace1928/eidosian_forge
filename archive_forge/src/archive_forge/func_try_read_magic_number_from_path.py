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
def try_read_magic_number_from_path(pathlike, count=8) -> bytes | None:
    if isinstance(pathlike, str) or hasattr(pathlike, '__fspath__'):
        path = os.fspath(pathlike)
        try:
            with open(path, 'rb') as f:
                return read_magic_number_from_file(f, count)
        except (FileNotFoundError, TypeError):
            pass
    return None