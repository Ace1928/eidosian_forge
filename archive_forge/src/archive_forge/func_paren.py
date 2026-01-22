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
@staticmethod
def paren(string):

    def all_in_parens(string):
        if string[0] != '(' or len(string) < 2:
            return False
        count = 1
        for i, char in enumerate(string[1:]):
            if char == '(':
                count += 1
            elif char == ')':
                count -= 1
            if count == 0 and i != len(string) - 2:
                return False
        assert count == 0
        return True
    if isinstance(string, CSEVariable) or re.match('^[a-z0-9_.]+$', string, re.I) or re.match('^\\([^)]*\\)$', string, re.I) or (string == ''):
        return string
    if all_in_parens(string):
        return string
    return f'({string})'