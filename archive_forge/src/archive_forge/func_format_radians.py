import re
from fractions import Fraction
from typing import (
import numpy as np
import sympy
from typing_extensions import Protocol
from cirq import protocols, value
from cirq._doc import doc_private
def format_radians(self, radians: Union[sympy.Basic, int, float]) -> str:
    """Returns angle in radians as a human-readable string."""
    if protocols.is_parameterized(radians):
        return str(radians)
    unit = 'π' if self.use_unicode_characters else 'pi'
    if radians == np.pi:
        return unit
    if radians == 0:
        return '0'
    if radians == -np.pi:
        return f'-{unit}'
    if self.precision is not None and (not isinstance(radians, sympy.Basic)):
        quantity = self.format_real(radians / np.pi)
        return quantity + unit
    return repr(radians)