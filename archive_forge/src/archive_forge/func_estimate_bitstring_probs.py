import sys
from collections import namedtuple
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, TYPE_CHECKING, Union, cast
import numpy as np
from pyquil.external.rpcq import CompilerISA
from pyquil.gates import I, RX, MEASURE
from pyquil.noise_gates import _get_qvm_noise_supported_gates
from pyquil.quilatom import MemoryReference, format_parameter, ParameterDesignator
from pyquil.quilbase import Pragma, Gate, Declare
def estimate_bitstring_probs(results: np.ndarray) -> np.ndarray:
    """
    Given an array of single shot results estimate the probability distribution over all bitstrings.

    :param results: A 2d array where the outer axis iterates over shots
        and the inner axis over bits.
    :return: An array with as many axes as there are qubit and normalized such that it sums to one.
        ``p[i,j,...,k]`` gives the estimated probability of bitstring ``ij...k``.
    """
    nshots, nq = np.shape(results)
    outcomes = np.array([int(''.join(map(str, r)), 2) for r in results])
    probs = np.histogram(outcomes, bins=np.arange(-0.5, 2 ** nq, 1))[0] / float(nshots)
    return _bitstring_probs_by_qubit(probs)