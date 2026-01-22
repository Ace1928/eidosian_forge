from sympy.core.backend import eye, Matrix, zeros
from sympy.physics.mechanics import dynamicsymbols
from sympy.physics.mechanics.functions import find_dynamicsymbols
@property
def comb_explicit_rhs(self):
    """Returns the right hand side of the equations of motion in explicit
        form, x' = F, where the kinematical equations are included"""
    if self._comb_explicit_rhs is None:
        raise AttributeError('Please run .combute_explicit_form before attempting to access comb_explicit_rhs.')
    else:
        return self._comb_explicit_rhs