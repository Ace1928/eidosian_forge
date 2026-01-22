from typing import Any, Callable, cast, Dict, Optional, Union
import numpy as np
import sympy
from cirq import ops
def ccz_to_quirk_op(gate: ops.CCZPowGate) -> Optional[QuirkOp]:
    e = angle_to_exponent_key(gate.exponent)
    if e is None:
        return None
    return QuirkOp('•', '•', 'Z' + e, can_merge=False)