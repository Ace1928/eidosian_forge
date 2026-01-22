from pythran.passmanager import ModuleAnalysis
import pythran.metadata as md
import gast as ast
def store_and_visit(self, node):
    self.expr_parent = node
    self.result[node] = self.locals.copy()
    self.generic_visit(node)