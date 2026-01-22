import sys
from collections import namedtuple
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, TYPE_CHECKING, Union, cast
import numpy as np
from pyquil.external.rpcq import CompilerISA
from pyquil.gates import I, RX, MEASURE
from pyquil.noise_gates import _get_qvm_noise_supported_gates
from pyquil.quilatom import MemoryReference, format_parameter, ParameterDesignator
from pyquil.quilbase import Pragma, Gate, Declare
def damping_kraus_map(p: float=0.1) -> List[np.ndarray]:
    """
    Generate the Kraus operators corresponding to an amplitude damping
    noise channel.

    :param p: The one-step damping probability.
    :return: A list [k1, k2] of the Kraus operators that parametrize the map.
    :rtype: list
    """
    damping_op = np.sqrt(p) * np.array([[0, 1], [0, 0]])
    residual_kraus = np.diag([1, np.sqrt(1 - p)])
    return [residual_kraus, damping_op]