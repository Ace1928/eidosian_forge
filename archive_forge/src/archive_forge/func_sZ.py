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
def sZ(q: int) -> PauliTerm:
    """
    A function that returns the sigma_Z operator on a particular qubit.

    :param qubit_index: The index of the qubit
    :returns: A PauliTerm object
    """
    return PauliTerm('Z', q)