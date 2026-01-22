from pythran import metadata
from pythran.analyses import HasBreak, HasContinue, NodeCount
from pythran.openmp import OMPDirective
from pythran.conversion import to_ast
from pythran.passmanager import Transformation
from copy import deepcopy
import gast as ast
def dc(body, i, n):
    if i == n - 1:
        return body
    else:
        return deepcopy(body)