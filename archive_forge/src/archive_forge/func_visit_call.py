from __future__ import annotations
import os
import typing as t
import astroid
from pylint.checkers import BaseChecker
def visit_call(self, node):
    """Visit a call node."""
    try:
        for i in node.func.inferred():
            func = None
            if isinstance(i, astroid.scoped_nodes.FunctionDef) and isinstance(i.parent, astroid.scoped_nodes.Module):
                func = '%s.%s' % (i.parent.name, i.name)
            if not func:
                continue
            entry = self.unwanted_functions.get(func)
            if entry and entry.applies_to(self.linter.current_file):
                self.add_message(self.BAD_FUNCTION, args=(entry.alternative, func), node=node)
    except astroid.exceptions.InferenceError:
        pass