from pythran import metadata
from pythran.analyses import HasBreak, HasContinue, NodeCount
from pythran.openmp import OMPDirective
from pythran.conversion import to_ast
from pythran.passmanager import Transformation
from copy import deepcopy
import gast as ast
def unroll(elt, body):
    return [ast.Assign([deepcopy(node.target)], elt, None)] + body