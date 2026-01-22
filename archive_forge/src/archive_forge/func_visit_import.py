from __future__ import annotations
import os
import typing as t
import astroid
from pylint.checkers import BaseChecker
def visit_import(self, node):
    """Visit an import node."""
    for name in node.names:
        self._check_import(node, name[0])