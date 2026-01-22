from pythran.analyses.aliases import Aliases
from pythran.analyses.argument_effects import ArgumentEffects
from pythran.analyses.identifiers import Identifiers
from pythran.analyses.pure_expressions import PureExpressions
from pythran.passmanager import FunctionAnalysis
from pythran.syntax import PythranSyntaxError
from pythran.utils import get_variable, isattr
import pythran.metadata as md
import pythran.openmp as openmp
import gast as ast
import sys
def modify(self, name):
    dead_vars = [var for var, deps in self.use.items() if name in deps]
    self.dead.update(dead_vars)
    for var in dead_vars:
        dead_aliases = [alias.id for alias in self.name_to_nodes[var] if isinstance(alias, ast.Name)]
        self.dead.update(dead_aliases)