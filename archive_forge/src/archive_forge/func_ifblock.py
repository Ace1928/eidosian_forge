from __future__ import annotations
from dataclasses import dataclass, field
import re
import codecs
import os
import typing as T
from .mesonlib import MesonException
from . import mlog
def ifblock(self) -> IfClauseNode:
    if_node = self.create_node(SymbolNode, self.previous)
    condition = self.statement()
    clause = self.create_node(IfClauseNode, condition)
    self.expect('eol')
    block = self.codeblock()
    clause.ifs.append(self.create_node(IfNode, clause, if_node, condition, block))
    self.elseifblock(clause)
    clause.elseblock = self.elseblock()
    clause.endif = self.create_node(SymbolNode, self.current)
    return clause