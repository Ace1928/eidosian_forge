from pyomo.core import (
from pyomo.core.expr import sqrt
from pyomo.gdp import Disjunct, Disjunction
import pyomo.network as ntwk
def instantiate_hierarchical_nested_model(m):
    """helper function to instantiate a nested version of the model with
    the Disjuncts and Disjunctions on blocks"""
    m.disj1 = Disjunct()
    m.disjunct_block.disj2 = Disjunct()
    m.disj1.c = Constraint(expr=sum((m.x[i] ** 2 for i in m.I)) <= 1)
    m.disjunct_block.disj2.c = Constraint(expr=sum(((3 - m.x[i]) ** 2 for i in m.I)) <= 1)
    m.disjunct_block.disj2.disjunction = Disjunction(expr=[[sum((m.x[i] ** 2 for i in m.I)) <= 1], [sum(((3 - m.x[i]) ** 2 for i in m.I)) <= 1]])
    m.disjunction_block.disjunction = Disjunction(expr=[m.disj1, m.disjunct_block.disj2])