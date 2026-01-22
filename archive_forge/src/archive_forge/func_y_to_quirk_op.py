from typing import Any, Callable, cast, Dict, Optional, Union
import numpy as np
import sympy
from cirq import ops
def y_to_quirk_op(gate: ops.YPowGate) -> QuirkOp:
    return xyz_to_quirk_op('y', gate)