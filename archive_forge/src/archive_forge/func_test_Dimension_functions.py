from sympy.physics.units.systems.si import dimsys_SI
from sympy.core.numbers import pi
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (acos, atan2, cos)
from sympy.physics.units.dimensions import Dimension
from sympy.physics.units.definitions.dimension_definitions import (
from sympy.physics.units import foot
from sympy.testing.pytest import raises
def test_Dimension_functions():
    raises(TypeError, lambda: dimsys_SI.get_dimensional_dependencies(cos(length)))
    raises(TypeError, lambda: dimsys_SI.get_dimensional_dependencies(acos(angle)))
    raises(TypeError, lambda: dimsys_SI.get_dimensional_dependencies(atan2(length, time)))
    raises(TypeError, lambda: dimsys_SI.get_dimensional_dependencies(log(length)))
    raises(TypeError, lambda: dimsys_SI.get_dimensional_dependencies(log(100, length)))
    raises(TypeError, lambda: dimsys_SI.get_dimensional_dependencies(log(length, 10)))
    assert dimsys_SI.get_dimensional_dependencies(pi) == {}
    assert dimsys_SI.get_dimensional_dependencies(cos(1)) == {}
    assert dimsys_SI.get_dimensional_dependencies(cos(angle)) == {}
    assert dimsys_SI.get_dimensional_dependencies(atan2(length, length)) == {}
    assert dimsys_SI.get_dimensional_dependencies(log(length / length, length / length)) == {}
    assert dimsys_SI.get_dimensional_dependencies(Abs(length)) == {length: 1}
    assert dimsys_SI.get_dimensional_dependencies(Abs(length / length)) == {}
    assert dimsys_SI.get_dimensional_dependencies(sqrt(-1)) == {}