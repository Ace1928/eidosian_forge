from __future__ import annotations
from .basic import Atom, Basic
from .sorting import ordered
from .evalf import EvalfMixin
from .function import AppliedUndef
from .singleton import S
from .sympify import _sympify, SympifyError
from .parameters import global_parameters
from .logic import fuzzy_bool, fuzzy_xor, fuzzy_and, fuzzy_not
from sympy.logic.boolalg import Boolean, BooleanAtom
from sympy.utilities.iterables import sift
from sympy.utilities.misc import filldedent
from .expr import Expr
from sympy.multipledispatch import dispatch
from .containers import Tuple
from .symbol import Symbol
def split_real_imag(expr):
    real_imag = lambda t: 'real' if is_extended_real(t, assumptions) else 'imag' if is_extended_real(I * t, assumptions) else None
    return sift(Add.make_args(expr), real_imag)