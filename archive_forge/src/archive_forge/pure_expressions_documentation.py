from pythran.analyses.aliases import Aliases
from pythran.analyses.argument_effects import ArgumentEffects
from pythran.analyses.global_effects import GlobalEffects
from pythran.analyses.pure_functions import PureFunctions
from pythran.passmanager import ModuleAnalysis
from pythran.intrinsic import Intrinsic
import gast as ast
Yields the set of pure expressions