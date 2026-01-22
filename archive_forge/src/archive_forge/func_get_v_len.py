from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.core.numbers import Integer
@require_all_args
def get_v_len(self):
    return self.v_steps + 1