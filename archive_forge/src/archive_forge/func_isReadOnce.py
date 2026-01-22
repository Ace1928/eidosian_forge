from pythran.analyses.aliases import Aliases
from pythran.analyses.argument_read_once import ArgumentReadOnce
from pythran.passmanager import NodeAnalysis
import gast as ast
def isReadOnce(f, i):
    return f in self.argument_read_once and self.argument_read_once[f][i] <= 1