import dataclasses
import datetime
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, TYPE_CHECKING, Union
import numpy as np
import sympy
from cirq import ops, protocols, value
from cirq._compat import proper_repr
from cirq.work.observable_settings import (
def means(self, *, atol: float=1e-08) -> np.ndarray:
    """Estimates of the means of the settings in this accumulator."""
    return np.asarray([self.mean(setting, atol=atol) for setting in self.simul_settings])