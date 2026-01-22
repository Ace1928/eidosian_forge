import pyomo.common.unittest as unittest
from pyomo.environ import ConcreteModel, Constraint, TransformationFactory, Var
import pyomo.core.expr as EXPR
from pyomo.repn import generate_standard_repn
Test for removing zero terms from linear constraints.