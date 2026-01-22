from typing import Optional, Tuple, cast
import numpy as np
import numpy.typing as npt
from cirq.ops import DensePauliString
from cirq import protocols
Returns the gate that acts on the ith qubit.