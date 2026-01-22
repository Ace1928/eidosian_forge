import time
from typing import Any, Dict, Union, Sequence, List, Tuple, TYPE_CHECKING, Optional, cast
import sympy
import numpy as np
import scipy.optimize
from cirq import circuits, ops, vis, study
from cirq._compat import proper_repr
@property
def confusion_matrices(self) -> Tuple[np.ndarray, ...]:
    """List of confusion matrices corresponding to `measure_qubits` qubit pattern."""
    return self._confusion_matrices