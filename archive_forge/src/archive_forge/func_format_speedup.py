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
def format_speedup(speedup, pvalue, is_correct=True, pvalue_threshold=0.1):
    if not is_correct:
        return 'ERROR'
    if pvalue > pvalue_threshold:
        return f'{speedup:.3f}x SAME'
    return f'{speedup:.3f}x p={pvalue:.2f}'