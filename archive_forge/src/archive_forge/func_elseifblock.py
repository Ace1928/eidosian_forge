from __future__ import annotations
from dataclasses import dataclass, field
import re
import codecs
import os
import typing as T
from .mesonlib import MesonException
from . import mlog
def elseifblock(self, clause: IfClauseNode) -> None:
    while self.accept('elif'):
        elif_ = self.create_node(SymbolNode, self.previous)
        s = self.statement()
        self.expect('eol')
        b = self.codeblock()
        clause.ifs.append(self.create_node(IfNode, s, elif_, s, b))