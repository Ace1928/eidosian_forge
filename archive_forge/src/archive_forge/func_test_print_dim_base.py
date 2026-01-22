from sympy.core.symbol import symbols
from sympy.matrices.dense import (Matrix, eye)
from sympy.physics.units.definitions.dimension_definitions import (
from sympy.physics.units.dimensions import DimensionSystem
def test_print_dim_base():
    mksa = DimensionSystem((length, time, mass, current), (action,), {action: {mass: 1, length: 2, time: -1}})
    L, M, T = symbols('L M T')
    assert mksa.print_dim_base(action) == L ** 2 * M / T