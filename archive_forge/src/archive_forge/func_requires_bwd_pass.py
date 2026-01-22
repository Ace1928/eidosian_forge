import contextlib
import dis
import functools
import logging
import os.path
import random
import re
import sys
import types
import unittest
from typing import List, Optional, Sequence, Union
from unittest.mock import patch
import torch
from torch import fx
from torch._dynamo.output_graph import OutputGraph
from . import config, eval_frame, optimize_assert, reset
from .bytecode_transformation import (
from .guards import CheckFunctionManager, GuardedCode
from .utils import same
def requires_bwd_pass(out):
    if isinstance(out, torch.Tensor):
        return out.requires_grad
    elif isinstance(out, (list, tuple)):
        return any((requires_bwd_pass(x) for x in out))
    elif out is None:
        return False
    elif isinstance(out, int):
        return False
    raise NotImplementedError("Don't know how to reduce", type(out))