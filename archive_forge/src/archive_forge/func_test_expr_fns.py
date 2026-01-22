from sympy.strategies.traverse import (
from sympy.strategies.rl import rebuild
from sympy.strategies.util import expr_fns
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.numbers import Integer
from sympy.core.singleton import S
from sympy.core.symbol import Str, Symbol
from sympy.abc import x, y, z
def test_expr_fns():
    expr = x + y ** 3
    e = bottom_up(lambda v: v + 1, expr_fns)(expr)
    b = bottom_up(lambda v: Basic.__new__(Add, v, S(1)), basic_fns)(expr)
    assert rebuild(b) == e