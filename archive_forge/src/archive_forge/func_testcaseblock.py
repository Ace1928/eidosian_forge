from __future__ import annotations
from dataclasses import dataclass, field
import re
import codecs
import os
import typing as T
from .mesonlib import MesonException
from . import mlog
def testcaseblock(self) -> TestCaseClauseNode:
    testcase = self.create_node(SymbolNode, self.previous)
    condition = self.statement()
    self.expect('eol')
    block = self.codeblock()
    endtestcase = SymbolNode(self.current)
    return self.create_node(TestCaseClauseNode, testcase, condition, block, endtestcase)