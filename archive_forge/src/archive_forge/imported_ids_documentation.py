from pythran.analyses.globals_analysis import Globals
from pythran.analyses.locals_analysis import Locals
from pythran.passmanager import NodeAnalysis
import pythran.metadata as md
import gast as ast
Gather ids referenced by a node and not declared locally.