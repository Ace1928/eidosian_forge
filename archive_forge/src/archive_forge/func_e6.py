from __future__ import annotations
from dataclasses import dataclass, field
import re
import codecs
import os
import typing as T
from .mesonlib import MesonException
from . import mlog
def e6(self) -> BaseNode:
    if self.accept('not'):
        operator = self.create_node(SymbolNode, self.previous)
        return self.create_node(NotNode, self.current, operator, self.e7())
    if self.accept('dash'):
        operator = self.create_node(SymbolNode, self.previous)
        return self.create_node(UMinusNode, self.current, operator, self.e7())
    return self.e7()