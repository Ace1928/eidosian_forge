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
def random_well_conditioned_matrix(*shape, dtype, device, mean=1.0, sigma=0.001):
    """
    Returns a random rectangular matrix (batch of matrices)
    with singular values sampled from a Gaussian with
    mean `mean` and standard deviation `sigma`.
    The smaller the `sigma`, the better conditioned
    the output matrix is.
    """
    primitive_dtype = {torch.float: torch.float, torch.double: torch.double, torch.cfloat: torch.float, torch.cdouble: torch.double}
    x = torch.rand(shape, dtype=dtype, device=device)
    m = x.size(-2)
    n = x.size(-1)
    u, _, vh = torch.linalg.svd(x, full_matrices=False)
    s = (torch.randn(*shape[:-2] + (min(m, n),), dtype=primitive_dtype[dtype], device=device) * sigma + mean).sort(-1, descending=True).values.to(dtype)
    return u * s.unsqueeze(-2) @ vh