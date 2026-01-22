from itertools import product
from typing import Tuple as tTuple
from sympy.core.add import Add
from sympy.core.cache import cacheit
from sympy.core.expr import Expr
from sympy.core.function import (Function, ArgumentIndexError, expand_log,
from sympy.core.logic import fuzzy_and, fuzzy_not, fuzzy_or
from sympy.core.mul import Mul
from sympy.core.numbers import Integer, Rational, pi, I, ImaginaryUnit
from sympy.core.parameters import global_parameters
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import Wild, Dummy
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.complexes import arg, unpolarify, im, re, Abs
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.ntheory import multiplicity, perfect_power
from sympy.ntheory.factor_ import factorint
def coeff_exp(term, x):
    coeff, exp = (S.One, S.Zero)
    for factor in Mul.make_args(term):
        if factor.has(x):
            base, exp = factor.as_base_exp()
            if base != x:
                try:
                    return term.leadterm(x)
                except ValueError:
                    return (term, S.Zero)
        else:
            coeff *= factor
    return (coeff, exp)