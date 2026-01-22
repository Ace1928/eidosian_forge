import itertools
import types
import warnings
from collections import defaultdict
from typing import (
import numpy as np
from rpcq.messages import NativeQuilMetadata, ParameterAref
from pyquil._parser.parser import run_parser
from pyquil._memory import Memory
from pyquil.gates import MEASURE, RESET, MOVE
from pyquil.noise import _check_kraus_ops, _create_kraus_pragmas, pauli_kraus_map
from pyquil.quilatom import (
from pyquil.quilbase import (
from pyquil.quiltcalibrations import (
def define_noisy_readout(self, qubit: Union[int, QubitPlaceholder], p00: float, p11: float) -> 'Program':
    """
        For this program define a classical bit flip readout error channel parametrized by
        ``p00`` and ``p11``. This models the effect of thermal noise that corrupts the readout
        signal **after** it has interrogated the qubit.

        :param qubit: The qubit with noisy readout.
        :param p00: The probability of obtaining the measurement result 0 given that the qubit
          is in state 0.
        :param p11: The probability of obtaining the measurement result 1 given that the qubit
          is in state 1.
        :return: The Program with an appended READOUT-POVM Pragma.
        """
    if not 0.0 <= p00 <= 1.0:
        raise ValueError('p00 must be in the interval [0,1].')
    if not 0.0 <= p11 <= 1.0:
        raise ValueError('p11 must be in the interval [0,1].')
    if not (isinstance(qubit, int) or isinstance(qubit, QubitPlaceholder)):
        raise TypeError('qubit must be a non-negative integer, or QubitPlaceholder.')
    if isinstance(qubit, int) and qubit < 0:
        raise ValueError('qubit cannot be negative.')
    p00 = float(p00)
    p11 = float(p11)
    aprobs = [p00, 1.0 - p11, 1.0 - p00, p11]
    aprobs_str = '({})'.format(' '.join((format_parameter(p) for p in aprobs)))
    pragma = Pragma('READOUT-POVM', [qubit], aprobs_str)
    return self.inst(pragma)