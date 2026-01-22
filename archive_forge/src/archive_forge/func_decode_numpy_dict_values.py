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
def decode_numpy_dict_values(attrs: Mapping[K, V]) -> dict[K, V]:
    """Convert attribute values from numpy objects to native Python objects,
    for use in to_dict
    """
    attrs = dict(attrs)
    for k, v in attrs.items():
        if isinstance(v, np.ndarray):
            attrs[k] = v.tolist()
        elif isinstance(v, np.generic):
            attrs[k] = v.item()
    return attrs