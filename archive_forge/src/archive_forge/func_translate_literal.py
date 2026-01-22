from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.assumptions.ask_generated import get_all_known_facts
from sympy.assumptions.assume import global_assumptions, AppliedPredicate
from sympy.assumptions.sathandlers import class_fact_registry
from sympy.core import oo
from sympy.logic.inference import satisfiable
from sympy.assumptions.cnf import CNF, EncodedCNF
def translate_literal(lit, delta):
    if lit > 0:
        return lit + delta
    else:
        return lit - delta