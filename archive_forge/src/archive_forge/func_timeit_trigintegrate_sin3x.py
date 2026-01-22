from sympy.core.symbol import Symbol
from sympy.functions.elementary.trigonometric import sin
from sympy.integrals.trigonometry import trigintegrate
def timeit_trigintegrate_sin3x():
    trigintegrate(sin(x) ** 3, x)