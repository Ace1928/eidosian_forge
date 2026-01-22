from typing import (
import numpy as np
from cirq import _compat, protocols, value
from cirq.ops import raw_types
@property
def confusion_map(self) -> Dict[Tuple[int, ...], np.ndarray]:
    return self._confusion_map