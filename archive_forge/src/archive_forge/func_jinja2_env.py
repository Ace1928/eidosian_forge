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
@functools.lru_cache(None)
def jinja2_env():
    try:
        import jinja2
        return jinja2.Environment(undefined=jinja2.StrictUndefined)
    except ImportError:
        return None