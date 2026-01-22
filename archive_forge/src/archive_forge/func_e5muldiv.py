from __future__ import annotations
from dataclasses import dataclass, field
import re
import codecs
import os
import typing as T
from .mesonlib import MesonException
from . import mlog
def e5muldiv(self) -> BaseNode:
    op_map = {'percent': 'mod', 'star': 'mul', 'fslash': 'div'}
    left = self.e6()
    while True:
        op = self.accept_any(tuple(op_map.keys()))
        if op:
            operator = self.create_node(SymbolNode, self.previous)
            left = self.create_node(ArithmeticNode, op_map[op], left, operator, self.e6())
        else:
            break
    return left