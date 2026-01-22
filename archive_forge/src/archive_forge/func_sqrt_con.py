import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.network import Port, Arc
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.gdp import Disjunct, Disjunction
from pyomo.core.base.units_container import pint_available, UnitsError
from pyomo.util.check_units import (
@m.Constraint(m.S)
def sqrt_con(m, i):
    return sqrt(m.v[i]) == sqrt(m.x[i] / m.t[i])