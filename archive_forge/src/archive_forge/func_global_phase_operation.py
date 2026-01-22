from typing import AbstractSet, Any, cast, Dict, Sequence, Tuple, Union, Optional, Collection
import numpy as np
import sympy
import cirq
from cirq import value, protocols
from cirq.ops import raw_types, controlled_gate, control_values as cv
from cirq.type_workarounds import NotImplementedType
def global_phase_operation(coefficient: 'cirq.TParamValComplex', atol: float=1e-08) -> 'cirq.GateOperation':
    """Creates an operation that represents a global phase on the state."""
    return GlobalPhaseGate(coefficient, atol)()