from pythran.passmanager import ModuleAnalysis
from pythran.conversion import demangle
import gast as ast
import math
def visit_And(self, node):
    self.result.add(('builtins', 'pythran', 'and'))
    self.generic_visit(node)