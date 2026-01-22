import re
from itertools import product
import numpy as np
import copy
from typing import (
from pyquil.quilatom import (
from .quil import Program
from .gates import H, RZ, RX, CNOT, X, PHASE, QUANTUM_GATES
from numbers import Number, Complex
from collections import OrderedDict
import warnings
def compact_str(self) -> str:
    """A string representation of the PauliSum that is more compact than ``str(pauli_sum)``

        >>> pauli_sum = 2.0 * sX(1)* sZ(2) + 1.5 * sY(2)
        >>> str(pauli_sum)
        >>> '2.0*X1*X2 + 1.5*Y2'
        >>> pauli_sum.compact_str()
        >>> '2.0*X1X2+1.5*Y2'
        """
    return '+'.join([term.compact_str() for term in self.terms])