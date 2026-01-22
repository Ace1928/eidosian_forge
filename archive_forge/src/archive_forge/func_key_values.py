from __future__ import annotations
from dataclasses import dataclass, field
import re
import codecs
import os
import typing as T
from .mesonlib import MesonException
from . import mlog
def key_values(self) -> ArgumentNode:
    s = self.statement()
    a = self.create_node(ArgumentNode, self.current)
    while not isinstance(s, EmptyNode):
        if self.accept('colon'):
            a.columns.append(self.create_node(SymbolNode, self.previous))
            a.set_kwarg_no_check(s, self.statement())
            if not self.accept('comma'):
                return a
            a.commas.append(self.create_node(SymbolNode, self.previous))
        else:
            raise ParseException('Only key:value pairs are valid in dict construction.', self.getline(), s.lineno, s.colno)
        s = self.statement()
    return a