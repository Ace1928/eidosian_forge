from __future__ import annotations
from dataclasses import dataclass, field
import re
import codecs
import os
import typing as T
from .mesonlib import MesonException
from . import mlog
def e7(self) -> BaseNode:
    left = self.e8()
    block_start = self.current
    if self.accept('lparen'):
        lpar = self.create_node(SymbolNode, block_start)
        args = self.args()
        self.block_expect('rparen', block_start)
        rpar = self.create_node(SymbolNode, self.previous)
        if not isinstance(left, IdNode):
            raise ParseException('Function call must be applied to plain id', self.getline(), left.lineno, left.colno)
        assert isinstance(left.value, str)
        left = self.create_node(FunctionNode, left, lpar, args, rpar)
    go_again = True
    while go_again:
        go_again = False
        if self.accept('dot'):
            go_again = True
            left = self.method_call(left)
        if self.accept('lbracket'):
            go_again = True
            left = self.index_call(left)
    return left