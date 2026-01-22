from sympy.simplify.epathtools import epath, EPath
from sympy.testing.pytest import raises
from sympy.core.numbers import E
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.abc import x, y, z, t
def test_epath_apply():
    expr = [((x, 1, t), 2), ((3, y, 4), z)]
    func = lambda expr: expr ** 2
    assert epath('/*', expr, list) == [[(x, 1, t), 2], [(3, y, 4), z]]
    assert epath('/*/[0]', expr, list) == [([x, 1, t], 2), ([3, y, 4], z)]
    assert epath('/*/[1]', expr, func) == [((x, 1, t), 4), ((3, y, 4), z ** 2)]
    assert epath('/*/[2]', expr, list) == expr
    assert epath('/*/[0]/int', expr, func) == [((x, 1, t), 2), ((9, y, 16), z)]
    assert epath('/*/[0]/Symbol', expr, func) == [((x ** 2, 1, t ** 2), 2), ((3, y ** 2, 4), z)]
    assert epath('/*/[0]/int[1:]', expr, func) == [((x, 1, t), 2), ((3, y, 16), z)]
    assert epath('/*/[0]/Symbol[1:]', expr, func) == [((x, 1, t ** 2), 2), ((3, y ** 2, 4), z)]
    assert epath('/Symbol', x + y + z + 1, func) == x ** 2 + y ** 2 + z ** 2 + 1
    assert epath('/*/*/Symbol', t + sin(x + 1) + cos(x + y + E), func) == t + sin(x ** 2 + 1) + cos(x ** 2 + y ** 2 + E)