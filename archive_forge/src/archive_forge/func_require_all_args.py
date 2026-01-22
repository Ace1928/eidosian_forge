from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.core.numbers import Integer
def require_all_args(f):

    def check(self, *args, **kwargs):
        for g in [self._v, self._v_min, self._v_max, self._v_steps]:
            if g is None:
                raise ValueError('PlotInterval is incomplete.')
        return f(self, *args, **kwargs)
    return check