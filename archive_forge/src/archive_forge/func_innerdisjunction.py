from pyomo.core import (
from pyomo.core.expr import sqrt
from pyomo.gdp import Disjunct, Disjunction
import pyomo.network as ntwk
@disjunct.Disjunction([0])
def innerdisjunction(b, i):
    return [b.innerdisjunct[0], b.innerdisjunct[1]]