from __future__ import annotations
import os
import typing as t
import astroid
from pylint.checkers import BaseChecker
def visit_importfrom(self, node):
    """Visit an import from node."""
    self._check_importfrom(node, node.modname, node.names)