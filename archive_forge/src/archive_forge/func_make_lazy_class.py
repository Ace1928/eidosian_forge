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
def make_lazy_class(cls):

    def lazy_init(self, cb):
        self._cb = cb
        self._value = None
    cls.__init__ = lazy_init
    for basename in ['add', 'sub', 'mul', 'truediv', 'floordiv', 'mod', 'divmod', 'pow', 'lshift', 'rshift', 'and', 'or', 'xor', 'neg', 'pos', 'abs', 'invert', 'eq', 'ne', 'lt', 'le', 'gt', 'ge', 'bool', 'int', 'index']:
        name = f'__{basename}__'

        def inner_wrapper(name):
            use_operator = basename not in ('bool', 'int')

            def wrapped(self, *args, **kwargs):
                if self._cb is not None:
                    self._value = self._cb()
                    self._cb = None
                if not use_operator:
                    return getattr(self._value, name)(*args, **kwargs)
                else:
                    return getattr(operator, name)(self._value, *args, **kwargs)
            return wrapped
        setattr(cls, name, inner_wrapper(name))
    return cls