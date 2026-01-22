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
def genSparseTensor(self, size, sparse_dim, nnz, is_uncoalesced, device, dtype):
    assert all((size[d] > 0 for d in range(sparse_dim))) or nnz == 0, 'invalid arguments'
    v_size = [nnz] + list(size[sparse_dim:])
    v = make_tensor(v_size, device=device, dtype=dtype, low=-1, high=1)
    i = torch.rand(sparse_dim, nnz, device=device)
    i.mul_(torch.tensor(size[:sparse_dim]).unsqueeze(1).to(i))
    i = i.to(torch.long)
    if is_uncoalesced:
        i1 = i[:, :nnz // 2, ...]
        i2 = i[:, :(nnz + 1) // 2, ...]
        i = torch.cat([i1, i2], 1)
    x = torch.sparse_coo_tensor(i, v, torch.Size(size), dtype=dtype, device=device)
    if not is_uncoalesced:
        x = x.coalesce()
    else:
        x = x.detach().clone()._coalesced_(False)
    return (x, x._indices().clone(), x._values().clone())