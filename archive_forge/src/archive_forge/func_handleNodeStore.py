import __future__
import builtins
import ast
import collections
import contextlib
import doctest
import functools
import os
import re
import string
import sys
import warnings
from pyflakes import messages
def handleNodeStore(self, node):
    name = getNodeName(node)
    if not name:
        return
    if isinstance(self.scope, FunctionScope) and name not in self.scope:
        for scope in self.scopeStack[:-1]:
            if not isinstance(scope, (FunctionScope, ModuleScope)):
                continue
            used = name in scope and scope[name].used
            if used and used[0] is self.scope and (name not in self.scope.globals):
                self.report(messages.UndefinedLocal, scope[name].used[1], name, scope[name].source)
                break
    parent_stmt = self.getParent(node)
    if isinstance(parent_stmt, ast.AnnAssign) and parent_stmt.value is None:
        binding = Annotation(name, node)
    elif isinstance(parent_stmt, (FOR_TYPES, ast.comprehension)) or (parent_stmt != node._pyflakes_parent and (not self.isLiteralTupleUnpacking(parent_stmt))):
        binding = Binding(name, node)
    elif name == '__all__' and isinstance(self.scope, ModuleScope) and isinstance(node._pyflakes_parent, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
        binding = ExportBinding(name, node._pyflakes_parent, self.scope)
    elif isinstance(parent_stmt, ast.NamedExpr):
        binding = NamedExprAssignment(name, node)
    else:
        binding = Assignment(name, node)
    self.addBinding(node, binding)