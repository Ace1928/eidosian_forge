from pythran.analyses.aliases import StrictAliases
from pythran.analyses.global_declarations import GlobalDeclarations
from pythran.passmanager import ModuleAnalysis
import gast as ast
Order all global functions according to their callgraph depth