from __future__ import annotations
from dataclasses import dataclass, field
import re
import codecs
import os
import typing as T
from .mesonlib import MesonException
from . import mlog
def index_call(self, source_object: BaseNode) -> IndexNode:
    lbracket = self.create_node(SymbolNode, self.previous)
    index_statement = self.statement()
    self.expect('rbracket')
    rbracket = self.create_node(SymbolNode, self.previous)
    return self.create_node(IndexNode, source_object, lbracket, index_statement, rbracket)