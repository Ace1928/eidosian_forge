from collections import defaultdict
from itertools import product
from functools import reduce
from math import prod
from sympy import SYMPY_DEBUG
from sympy.core import (S, Dummy, symbols, sympify, Tuple, expand, I, pi, Mul,
from sympy.core.mod import Mod
from sympy.core.sorting import default_sort_key
from sympy.functions import (exp, sqrt, root, log, lowergamma, cos,
from sympy.functions.elementary.complexes import polarify, unpolarify
from sympy.functions.special.hyper import (hyper, HyperRep_atanh,
from sympy.matrices import Matrix, eye, zeros
from sympy.polys import apart, poly, Poly
from sympy.series import residue
from sympy.simplify.powsimp import powdenest
from sympy.utilities.iterables import sift
def detect_3113(func):
    """https://functions.wolfram.com/07.34.03.0984.01"""
    x = func.an[0]
    u, v, w = func.bm
    if _mod1((u - v).simplify()) == 0:
        if _mod1((v - w).simplify()) == 0:
            return
        sig = (S.Half, S.Half, S.Zero)
        x1, x2, y = (u, v, w)
    elif _mod1((x - u).simplify()) == 0:
        sig = (S.Half, S.Zero, S.Half)
        x1, y, x2 = (u, v, w)
    else:
        sig = (S.Zero, S.Half, S.Half)
        y, x1, x2 = (u, v, w)
    if _mod1((x - x1).simplify()) != 0 or _mod1((x - x2).simplify()) != 0 or _mod1((x - y).simplify()) != S.Half or (x - x1 > 0) or (x - x2 > 0):
        return
    return ({a: x}, G_Function([x], [], [x - S.Half + t for t in sig], []))