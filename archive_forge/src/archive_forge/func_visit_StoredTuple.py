from pythran.analyses.globals_analysis import Globals
from pythran.analyses.locals_analysis import Locals
from pythran.passmanager import NodeAnalysis
import pythran.metadata as md
import gast as ast
def visit_StoredTuple(self, node):
    for elt in node.elts:
        if isinstance(elt, ast.Name):
            self.current_locals.add(elt.id)
            continue
        if isinstance(elt, ast.Subscript):
            self.visit(elt)
        if isinstance(elt, ast.Tuple):
            self.visit_StoredTuple(node)