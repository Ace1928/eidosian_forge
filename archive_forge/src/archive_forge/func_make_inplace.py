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
def make_inplace(self, input_name, output_name):
    assert output_name not in self.inplace_buffers
    if input_name in self.inplace_buffers:
        buf = self.inplace_buffers[input_name]
        buf.other_names.append(output_name)
        self.inplace_buffers[output_name] = buf
    else:
        buf = InplacedBuffer(f'in_out_ptr{len(unique(self.inplace_buffers.values()))}', [input_name, output_name])
        self.inplace_buffers[input_name] = buf
        self.inplace_buffers[output_name] = buf