from itertools import combinations, product, zip_longest
from sympy.assumptions.assume import AppliedPredicate, Predicate
from sympy.core.relational import Eq, Ne, Gt, Lt, Ge, Le
from sympy.core.singleton import S
from sympy.logic.boolalg import Or, And, Not, Xnor
from sympy.logic.boolalg import (Equivalent, ITE, Implies, Nand, Nor, Xor)
def from_cnf(self, cnf):
    self._symbols = list(cnf.all_predicates())
    n = len(self._symbols)
    self.encoding = dict(zip(self._symbols, range(1, n + 1)))
    self.data = [self.encode(clause) for clause in cnf.clauses]