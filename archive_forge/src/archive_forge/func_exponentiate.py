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
def exponentiate(term: PauliTerm) -> Program:
    """
    Creates a pyQuil program that simulates the unitary evolution exp(-1j * term)

    :param term: A pauli term to exponentiate
    :returns: A Program object
    """
    return exponential_map(term)(1.0)