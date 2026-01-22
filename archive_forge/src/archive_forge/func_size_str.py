import copy
import functools
import itertools
import multiprocessing.pool
import os
import queue
import re
import types
import warnings
from contextlib import contextmanager
from dataclasses import fields, is_dataclass
from multiprocessing import Manager
from queue import Empty
from shutil import disk_usage
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, TypeVar, Union
from urllib.parse import urlparse
import multiprocess
import multiprocess.pool
import numpy as np
from tqdm.auto import tqdm
from .. import config
from ..parallel import parallel_map
from . import logging
from . import tqdm as hf_tqdm
from ._dill import (  # noqa: F401 # imported for backward compatibility. TODO: remove in 3.0.0
def size_str(size_in_bytes):
    """Returns a human readable size string.

    If size_in_bytes is None, then returns "Unknown size".

    For example `size_str(1.5 * datasets.units.GiB) == "1.50 GiB"`.

    Args:
        size_in_bytes: `int` or `None`, the size, in bytes, that we want to
            format as a human-readable size string.
    """
    if not size_in_bytes:
        return 'Unknown size'
    _NAME_LIST = [('PiB', 2 ** 50), ('TiB', 2 ** 40), ('GiB', 2 ** 30), ('MiB', 2 ** 20), ('KiB', 2 ** 10)]
    size_in_bytes = float(size_in_bytes)
    for name, size_bytes in _NAME_LIST:
        value = size_in_bytes / size_bytes
        if value >= 1.0:
            return f'{value:.2f} {name}'
    return f'{int(size_in_bytes)} bytes'