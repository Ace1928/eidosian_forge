from __future__ import annotations
import os
import typing as t
import astroid
from pylint.checkers import BaseChecker
def visit_attribute(self, node):
    """Visit an attribute node."""
    last_child = node.last_child()
    if not isinstance(last_child, astroid.node_classes.Name):
        return
    module = last_child.name
    entry = self.unwanted_imports.get(module)
    if entry and entry.names:
        if entry.applies_to(self.linter.current_file, node.attrname):
            self.add_message(self.BAD_IMPORT_FROM, args=(node.attrname, entry.alternative, module), node=node)