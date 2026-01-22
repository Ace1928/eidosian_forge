import gast as ast
import itertools
import os
from pythran.analyses import GlobalDeclarations
from pythran.errors import PythranInternalError
from pythran.passmanager import ModuleAnalysis
from pythran.types.conversion import PYTYPE_TO_CTYPE_TABLE
from pythran.utils import get_variable
from pythran.typing import List, Set, Dict, NDArray, Tuple, Pointer, Fun
from pythran.graph import DiGraph
def visit_any_conditionnal(self, node1, node2):
    """
        Set and restore the in_cond variable before visiting subnode.

        Compute correct dependencies on a value as both branch are possible
        path.
        """
    true_naming = false_naming = None
    try:
        tmp = self.naming.copy()
        for expr in node1:
            self.visit(expr)
        true_naming = self.naming
        self.naming = tmp
    except KeyError:
        pass
    try:
        tmp = self.naming.copy()
        for expr in node2:
            self.visit(expr)
        false_naming = self.naming
        self.naming = tmp
    except KeyError:
        pass
    if true_naming and (not false_naming):
        self.naming = true_naming
    elif false_naming and (not true_naming):
        self.naming = false_naming
    elif true_naming and false_naming:
        self.naming = false_naming
        for k, v in true_naming.items():
            if k not in self.naming:
                self.naming[k] = v
            else:
                for dep in v:
                    if dep not in self.naming[k]:
                        self.naming[k].append(dep)