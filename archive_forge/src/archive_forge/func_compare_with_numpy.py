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
def compare_with_numpy(self, torch_fn, np_fn, tensor_like, device=None, dtype=None, **kwargs):
    assert TEST_NUMPY
    if isinstance(tensor_like, torch.Tensor):
        assert device is None
        assert dtype is None
        t_cpu = tensor_like.detach().cpu()
        if t_cpu.dtype is torch.bfloat16:
            t_cpu = t_cpu.float()
        a = t_cpu.numpy()
        t = tensor_like
    else:
        d = copy.copy(torch_to_numpy_dtype_dict)
        d[torch.bfloat16] = np.float32
        a = np.array(tensor_like, dtype=d[dtype])
        t = torch.tensor(tensor_like, device=device, dtype=dtype)
    np_result = np_fn(a)
    torch_result = torch_fn(t).cpu()
    if isinstance(np_result, np.ndarray):
        try:
            np_result = torch.from_numpy(np_result)
        except Exception:
            np_result = torch.from_numpy(np_result.copy())
        if t.dtype is torch.bfloat16 and torch_result.dtype is torch.bfloat16 and (np_result.dtype is torch.float):
            torch_result = torch_result.to(torch.float)
    self.assertEqual(np_result, torch_result, **kwargs)