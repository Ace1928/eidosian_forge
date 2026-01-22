import contextlib
import dataclasses
import functools
import itertools
import logging
import operator
import re
from collections import namedtuple
from itertools import chain
from typing import (
import sympy
from sympy.printing.printer import Printer
import torch
import torch.fx
from torch.utils._sympy.value_ranges import ValueRanges
from .. import config, metrics
from ..utils import (
from ..virtualized import ops, OpsValue, V
@contextlib.contextmanager
def swap_buffers(self, lb, cb=None, sb=None):
    if cb is None:
        cb = lb
    loads = self.loads
    compute = self.compute
    stores = self.stores
    cse = self.cse
    self.loads = lb
    self.compute = cb
    self.stores = sb
    self.cse = cse.clone()
    try:
        yield
    finally:
        self.loads = loads
        self.compute = compute
        self.stores = stores
        self.cse = cse