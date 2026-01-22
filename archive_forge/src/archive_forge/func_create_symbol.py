from abc import ABC, abstractmethod
from sympy.core.backend import pi, AppliedUndef, Derivative, Matrix
from sympy.physics.mechanics.body import Body
from sympy.physics.mechanics.functions import _validate_coordinates
from sympy.physics.vector import (Vector, dynamicsymbols, cross, Point,
from sympy.utilities.iterables import iterable
from sympy.utilities.exceptions import sympy_deprecation_warning
def create_symbol(number):
    if n_coords == 1 and (not number_single):
        return dynamicsymbols(f'{label}_{self.name}')
    return dynamicsymbols(f'{label}{number}_{self.name}')