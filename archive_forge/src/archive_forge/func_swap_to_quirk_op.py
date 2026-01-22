from typing import Any, Callable, cast, Dict, Optional, Union
import numpy as np
import sympy
from cirq import ops
def swap_to_quirk_op(gate: ops.SwapPowGate) -> Optional[QuirkOp]:
    if gate.exponent == 1:
        return QuirkOp('Swap', 'Swap', can_merge=False)
    return None