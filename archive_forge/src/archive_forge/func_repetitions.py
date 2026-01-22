import time
from typing import Any, Dict, Union, Sequence, List, Tuple, TYPE_CHECKING, Optional, cast
import sympy
import numpy as np
import scipy.optimize
from cirq import circuits, ops, vis, study
from cirq._compat import proper_repr
@property
def repetitions(self) -> int:
    """The number of repetitions that were used to estimate the confusion matrices."""
    return self._repetitions