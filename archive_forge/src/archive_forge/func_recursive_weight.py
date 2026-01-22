from pythran.analyses.aliases import Aliases
from pythran.analyses.global_declarations import GlobalDeclarations
from pythran.passmanager import ModuleAnalysis
from pythran.tables import MODULES
import pythran.intrinsic as intrinsic
import gast as ast
from functools import reduce
def recursive_weight(self, function, index, predecessors):
    if len(function.read_effects) <= index:
        return 0
    if function.read_effects[index] == -1:
        cycle = function in predecessors
        predecessors.add(function)
        if cycle:
            function.read_effects[index] = 2 * function.dependencies(ArgumentReadOnce.Context(function, index, predecessors, False))
        else:
            function.read_effects[index] = function.dependencies(ArgumentReadOnce.Context(function, index, predecessors, True))
    return function.read_effects[index]