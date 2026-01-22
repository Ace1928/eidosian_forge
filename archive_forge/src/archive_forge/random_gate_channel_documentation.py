import numbers
from typing import AbstractSet, Tuple, TYPE_CHECKING, Dict, Any, cast, SupportsFloat, Optional
import numpy as np
from cirq import protocols, value
from cirq._compat import proper_repr
from cirq.ops import raw_types
Applies a sub gate with some probability.