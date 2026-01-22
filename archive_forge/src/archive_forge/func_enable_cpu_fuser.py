from torch.autograd import Variable
from torch.autograd.function import _nested_map
from torch.jit.annotations import BroadcastingList2, BroadcastingList3  # noqa: F401
from torch.onnx import OperatorExportTypes
import torch
import torch.cuda
import torch.jit
import torch.jit._logging
import torch.jit.frontend
import torch.jit.quantized
import zipfile
import functools
from torch.testing import FileCheck
from torch.testing._internal.common_utils import IS_WINDOWS, \
from torch.testing._internal.common_jit import JitCommonTestCase
from torch.testing._internal.common_utils import enable_profiling_mode  # noqa: F401
from contextlib import contextmanager
from functools import reduce
from io import StringIO
from collections import defaultdict
import importlib.util
import inspect
import io
import math
import os
import pickle
import sys
import tempfile
import textwrap
from importlib.abc import Loader
from typing import Any, Dict, List, Tuple, Union
def enable_cpu_fuser(fn):

    def wrapper(*args, **kwargs):
        torch._C._jit_override_can_fuse_on_cpu_legacy(True)
        torch._C._jit_override_can_fuse_on_cpu(True)
        torch._C._jit_set_te_must_use_llvm_cpu(False)
        try:
            fn(*args, **kwargs)
        finally:
            torch._C._jit_override_can_fuse_on_cpu_legacy(False)
            torch._C._jit_override_can_fuse_on_cpu(False)
            torch._C._jit_set_te_must_use_llvm_cpu(True)
    return wrapper