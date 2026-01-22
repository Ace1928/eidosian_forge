from collections import defaultdict, OrderedDict
from contextlib import contextmanager
import sys
import gast as ast
def parentFunction(self, node):
    return self.parentInstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))