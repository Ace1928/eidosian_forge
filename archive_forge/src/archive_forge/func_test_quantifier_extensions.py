import contextlib
import itertools
import re
import typing
from enum import Enum
from typing import Callable
import sympy
from sympy import Add, Implies, sqrt
from sympy.core import Mul, Pow
from sympy.core import (S, pi, symbols, Function, Rational, Integer,
from sympy.functions import Piecewise, exp, sin, cos
from sympy.printing.smtlib import smtlib_code
from sympy.testing.pytest import raises, Failed
def test_quantifier_extensions():
    from sympy.logic.boolalg import Boolean
    from sympy import Interval, Tuple, sympify

    class ForAll(Boolean):

        def _smtlib(self, printer):
            bound_symbol_declarations = [printer._s_expr(sym.name, [printer._known_types[printer.symbol_table[sym]], Interval(start, end)]) for sym, start, end in self.limits]
            return printer._s_expr('forall', [printer._s_expr('', bound_symbol_declarations), self.function])

        @property
        def bound_symbols(self):
            return {s for s, _, _ in self.limits}

        @property
        def free_symbols(self):
            bound_symbol_names = {s.name for s in self.bound_symbols}
            return {s for s in self.function.free_symbols if s.name not in bound_symbol_names}

        def __new__(cls, *args):
            limits = [sympify(a) for a in args if isinstance(a, tuple) or isinstance(a, Tuple)]
            function = [sympify(a) for a in args if isinstance(a, Boolean)]
            assert len(limits) + len(function) == len(args)
            assert len(function) == 1
            function = function[0]
            if isinstance(function, ForAll):
                return ForAll.__new__(ForAll, *limits + function.limits, function.function)
            inst = Boolean.__new__(cls)
            inst._args = tuple(limits + [function])
            inst.limits = limits
            inst.function = function
            return inst
    f = Function('f')
    with _check_warns([_W.DEFAULTING_TO_FLOAT]) as w:
        assert smtlib_code(ForAll((x, -42, +21), Eq(f(x), f(x))), symbol_table={f: Callable[[float], float]}, log_warn=w) == '(assert (forall ( (x Real [-42, 21])) true))'
    with _check_warns([_W.DEFAULTING_TO_FLOAT] * 2) as w:
        assert smtlib_code(ForAll((x, -42, +21), (y, -100, 3), Implies(Eq(x, y), Eq(f(x), f(y)))), symbol_table={f: Callable[[float], float]}, log_warn=w) == '(declare-fun f (Real) Real)\n(assert (forall ( (x Real [-42, 21]) (y Real [-100, 3])) (=> (= x y) (= (f x) (f y)))))'
    a = Symbol('a', integer=True)
    b = Symbol('b', real=True)
    c = Symbol('c')
    with _check_warns([]) as w:
        assert smtlib_code(ForAll((a, 2, 100), ForAll((b, 2, 100), Implies(a < b, sqrt(a) < b) | c)), log_warn=w) == '(declare-const c Bool)\n(assert (forall ( (a Int [2, 100]) (b Real [2, 100])) (or c (=> (< a b) (< (pow a (/ 1 2)) b)))))'