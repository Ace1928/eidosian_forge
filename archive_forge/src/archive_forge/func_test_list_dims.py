from sympy.core.symbol import symbols
from sympy.matrices.dense import (Matrix, eye)
from sympy.physics.units.definitions.dimension_definitions import (
from sympy.physics.units.dimensions import DimensionSystem
def test_list_dims():
    dimsys = DimensionSystem((length, time, mass))
    assert dimsys.list_can_dims == (length, mass, time)