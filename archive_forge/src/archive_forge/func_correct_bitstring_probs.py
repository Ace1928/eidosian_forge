import sys
from collections import namedtuple
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, TYPE_CHECKING, Union, cast
import numpy as np
from pyquil.external.rpcq import CompilerISA
from pyquil.gates import I, RX, MEASURE
from pyquil.noise_gates import _get_qvm_noise_supported_gates
from pyquil.quilatom import MemoryReference, format_parameter, ParameterDesignator
from pyquil.quilbase import Pragma, Gate, Declare
def correct_bitstring_probs(p: np.ndarray, assignment_probabilities: List[np.ndarray]) -> np.ndarray:
    """
    Given a 2d array of corrupted bitstring probabilities (outer axis iterates over shots, inner
    axis over bits) and a list of assignment probability matrices (one for each bit in the readout)
    compute the corrected probabilities.

    :param p: An array that enumerates bitstring probabilities. When
        flattened out ``p = [p_00...0, p_00...1, ...,p_11...1]``. The total number of elements must
        therefore be a power of 2. The canonical shape has a separate axis for each qubit, such that
        ``p[i,j,...,k]`` gives the estimated probability of bitstring ``ij...k``.
    :param assignment_probabilities: A list of assignment probability matrices
        per qubit. Each assignment probability matrix is expected to be of the form::

            [[p00 p01]
             [p10 p11]]

    :return: ``p_corrected`` an array with as many dimensions as there are qubits that contains
        the noisy-readout-corrected estimated probabilities for each measured bitstring, i.e.,
        ``p[i,j,...,k]`` gives the estimated probability of bitstring ``ij...k``.
    """
    return _apply_local_transforms(p, (np.linalg.inv(ap) for ap in assignment_probabilities))