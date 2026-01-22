from typing import Any, Callable, cast, Dict, Optional, Union
import numpy as np
import sympy
from cirq import ops
def z_to_quirk_op(gate: ops.ZPowGate) -> QuirkOp:
    return xyz_to_quirk_op('z', gate)