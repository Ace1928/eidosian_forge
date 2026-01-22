from __future__ import annotations
import atexit
import builtins
import io
import logging
import math
import os
import re
import struct
import sys
import tempfile
import warnings
from collections.abc import Callable, MutableMapping
from enum import IntEnum
from pathlib import Path
from . import (
from ._binary import i32le, o32be, o32le
from ._util import DeferredError, is_path
def convert_transparency(m, v):
    v = m[0] * v[0] + m[1] * v[1] + m[2] * v[2] + m[3] * 0.5
    return max(0, min(255, int(v)))