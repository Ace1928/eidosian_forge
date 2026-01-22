from sympy.core import Basic, Expr
from sympy.core.function import Lambda
from sympy.core.numbers import oo, Infinity, NegativeInfinity, Zero, Integer
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.miscellaneous import (Max, Min)
from sympy.sets.fancysets import ImageSet
from sympy.sets.setexpr import set_div
from sympy.sets.sets import Set, Interval, FiniteSet, Union
from sympy.multipledispatch import Dispatcher

    Powers in interval arithmetic
    https://en.wikipedia.org/wiki/Interval_arithmetic
    