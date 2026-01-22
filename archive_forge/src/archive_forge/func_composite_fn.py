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
def composite_fn(test, generic_cls, device_cls, old_parametrize_fn=old_parametrize_fn, new_parametrize_fn=new_parametrize_fn):
    old_tests = list(old_parametrize_fn(test, generic_cls, device_cls))
    for old_test, old_test_name, old_param_kwargs, old_dec_fn in old_tests:
        for new_test, new_test_name, new_param_kwargs, new_dec_fn in new_parametrize_fn(old_test, generic_cls, device_cls):
            redundant_params = set(old_param_kwargs.keys()).intersection(new_param_kwargs.keys())
            if redundant_params:
                raise RuntimeError('Parametrization over the same parameter by multiple parametrization decorators is not supported. For test "{}", the following parameters are handled multiple times: {}'.format(test.__name__, redundant_params))
            full_param_kwargs = {**old_param_kwargs, **new_param_kwargs}
            merged_test_name = '{}{}{}'.format(new_test_name, '_' if old_test_name != '' and new_test_name != '' else '', old_test_name)

            def merged_decorator_fn(param_kwargs, old_dec_fn=old_dec_fn, new_dec_fn=new_dec_fn):
                return list(old_dec_fn(param_kwargs)) + list(new_dec_fn(param_kwargs))
            yield (new_test, merged_test_name, full_param_kwargs, merged_decorator_fn)