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
def newvar(self, bounds: ValueRanges=ValueRanges.unknown()) -> CSEVariable:
    var_name = f'{self.name_prefix}{next(self.iter_buffer_ids)}'
    var = V.kernel.create_cse_var(var_name, bounds)
    self.varname_map[var_name] = var
    return var