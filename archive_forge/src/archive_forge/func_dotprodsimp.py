from contextlib import contextmanager
from threading import local
from sympy.core.function import expand_mul
@contextmanager
def dotprodsimp(x):
    old = _dotprodsimp_state.state
    try:
        _dotprodsimp_state.state = x
        yield
    finally:
        _dotprodsimp_state.state = old