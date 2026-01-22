import math
from ..util.pyutil import deprecated
from ..util._expr import Expr
def equilibrium_equation(self, variables, backend=math, equilibrium=None, **kwargs):
    return self.eq_const(variables, backend=backend, **kwargs) - self.active_conc_prod(variables, backend=backend, equilibrium=equilibrium, **kwargs)