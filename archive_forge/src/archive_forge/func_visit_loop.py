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
def visit_loop(self, body):
    old_pre_count = self.pre_loop_count
    self.pre_loop_count = dict()
    for stmt in body:
        self.visit(stmt)
    no_assign = [n for n, (_, a) in self.pre_loop_count.items() if not a]
    self.result.update(zip(no_assign, [LazynessAnalysis.MANY] * len(no_assign)))
    for k, v in self.pre_loop_count.items():
        loop_value = v[0] + self.name_count[k]
        self.result[k] = max(self.result.get(k, 0), loop_value)
    dead = self.dead.intersection(self.pre_loop_count)
    self.result.update(zip(dead, [LazynessAnalysis.INF] * len(dead)))
    for k, v in old_pre_count.items():
        if v[1] or k not in self.pre_loop_count:
            self.pre_loop_count[k] = v
        else:
            self.pre_loop_count[k] = (v[0] + self.pre_loop_count[k][0], self.pre_loop_count[k][1])