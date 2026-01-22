from pythran.analyses.aliases import Aliases
from pythran.analyses.global_declarations import GlobalDeclarations
from pythran.passmanager import ModuleAnalysis
from pythran.tables import MODULES
import pythran.intrinsic as intrinsic
import gast as ast
from functools import reduce
def save_effect(module):
    """ Recursively save read once effect for Pythonic functions. """
    for intr in module.values():
        if isinstance(intr, dict):
            save_effect(intr)
        else:
            fe = ArgumentReadOnce.FunctionEffects(intr)
            self.node_to_functioneffect[intr] = fe
            self.result.add(fe)
            if isinstance(intr, intrinsic.Class):
                save_effect(intr.fields)