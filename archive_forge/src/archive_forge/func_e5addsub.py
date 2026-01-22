from __future__ import annotations
from dataclasses import dataclass, field
import re
import codecs
import os
import typing as T
from .mesonlib import MesonException
from . import mlog
def e5addsub(self) -> BaseNode:
    op_map = {'plus': 'add', 'dash': 'sub'}
    left = self.e5muldiv()
    while True:
        op = self.accept_any(tuple(op_map.keys()))
        if op:
            operator = self.create_node(SymbolNode, self.previous)
            left = self.create_node(ArithmeticNode, op_map[op], left, operator, self.e5muldiv())
        else:
            break
    return left