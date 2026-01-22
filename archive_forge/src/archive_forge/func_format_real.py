import re
from fractions import Fraction
from typing import (
import numpy as np
import sympy
from typing_extensions import Protocol
from cirq import protocols, value
from cirq._doc import doc_private
def format_real(self, val: Union[sympy.Basic, int, float]) -> str:
    if isinstance(val, sympy.Basic):
        return str(val)
    if val == int(val):
        return str(int(val))
    if self.precision is None:
        return str(val)
    return f'{float(val):.{self.precision}}'