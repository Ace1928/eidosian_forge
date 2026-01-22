import argparse
import contextlib
import copy
import ctypes
import errno
import functools
import gc
import inspect
import io
import json
import logging
import math
import operator
import os
import platform
import random
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
import warnings
from collections.abc import Mapping, Sequence
from contextlib import closing, contextmanager
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from functools import partial, wraps
from itertools import product, chain
from pathlib import Path
from statistics import mean
from typing import (
from unittest.mock import MagicMock
import expecttest
import numpy as np
import __main__  # type: ignore[import]
import torch
import torch.backends.cudnn
import torch.backends.mkl
import torch.backends.mps
import torch.backends.xnnpack
import torch.cuda
from torch import Tensor
from torch._C import ScriptDict, ScriptList  # type: ignore[attr-defined]
from torch._utils_internal import get_writable_path
from torch.nn import (
from torch.onnx import (
from torch.testing import make_tensor
from torch.testing._comparison import (
from torch.testing._comparison import not_close_error_metas
from torch.testing._internal.common_dtype import get_all_dtypes
import torch.utils._pytree as pytree
from .composite_compliance import no_dispatch
def bytes_to_scalar(byte_list: List[int], dtype: torch.dtype, device: torch.device):
    dtype_to_ctype: Dict[torch.dtype, Any] = {torch.int8: ctypes.c_int8, torch.uint8: ctypes.c_uint8, torch.int16: ctypes.c_int16, torch.int32: ctypes.c_int32, torch.int64: ctypes.c_int64, torch.bool: ctypes.c_bool, torch.float32: ctypes.c_float, torch.complex64: ctypes.c_float, torch.float64: ctypes.c_double, torch.complex128: ctypes.c_double}
    ctype = dtype_to_ctype[dtype]
    num_bytes = ctypes.sizeof(ctype)

    def check_bytes(byte_list):
        for byte in byte_list:
            assert 0 <= byte <= 255
    if dtype.is_complex:
        assert len(byte_list) == num_bytes * 2
        check_bytes(byte_list)
        real = ctype.from_buffer((ctypes.c_byte * num_bytes)(*byte_list[:num_bytes])).value
        imag = ctype.from_buffer((ctypes.c_byte * num_bytes)(*byte_list[num_bytes:])).value
        res = real + 1j * imag
    else:
        assert len(byte_list) == num_bytes
        check_bytes(byte_list)
        res = ctype.from_buffer((ctypes.c_byte * num_bytes)(*byte_list)).value
    return torch.tensor(res, device=device, dtype=dtype)