from typing import AbstractSet, Any, Dict, Optional, Tuple, TYPE_CHECKING, Union, List
import datetime
import sympy
import numpy as np
from cirq import protocols
from cirq._compat import proper_repr, cached_method
from cirq._doc import document
@cached_method
def total_picos(self) -> _NUMERIC_OUTPUT_TYPE:
    """Returns the number of picoseconds that the duration spans."""
    val = sum((a * b for a, b in zip(self._time_vals, self._multipliers)))
    return float(val) if isinstance(val, np.number) else val