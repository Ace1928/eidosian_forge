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
def term_with_coeff(term: PauliTerm, coeff: ExpressionDesignator) -> PauliTerm:
    """
    Change the coefficient of a PauliTerm.

    :param term: A PauliTerm object
    :param coeff: The coefficient to set on the PauliTerm
    :returns: A new PauliTerm that duplicates term but sets coeff
    """
    if not isinstance(coeff, Number):
        raise ValueError('coeff must be a Number')
    new_pauli = term.copy()
    new_pauli.coefficient = complex(coeff)
    return new_pauli