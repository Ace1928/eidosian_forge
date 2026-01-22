from typing import (
import numbers
import sympy
from cirq import value, protocols
from cirq._compat import proper_repr
from cirq.ops import (
@property
def exponent_neg(self) -> Union[int, float, sympy.Expr]:
    """The negative exponent."""
    return self._exponent_neg