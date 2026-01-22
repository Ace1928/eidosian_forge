from sympy.core.backend import eye, Matrix, zeros
from sympy.physics.mechanics import dynamicsymbols
from sympy.physics.mechanics.functions import find_dynamicsymbols
def dynamic_symbols(self):
    """Returns a column matrix containing all of the symbols in the system
        that depend on time"""
    if self._comb_explicit_rhs is None:
        eom_expressions = self.comb_implicit_mat[:] + self.comb_implicit_rhs[:]
    else:
        eom_expressions = self._comb_explicit_rhs[:]
    functions_of_time = set()
    for expr in eom_expressions:
        functions_of_time = functions_of_time.union(find_dynamicsymbols(expr))
    functions_of_time = functions_of_time.union(self._states)
    return tuple(functions_of_time)