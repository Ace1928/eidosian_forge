import itertools
from typing import Any, Dict, Optional, Sequence, Type, Union
import numpy as np
import sympy
from cirq import ops, protocols, value
from cirq.testing.consistent_act_on import assert_all_implemented_act_on_effects_match_unitary
from cirq.testing.circuit_compare import (
from cirq.testing.consistent_decomposition import (
from cirq.testing.consistent_phase_by import assert_phase_by_is_consistent_with_unitary
from cirq.testing.consistent_qasm import assert_qasm_is_consistent_with_unitary
from cirq.testing.consistent_pauli_expansion import (
from cirq.testing.consistent_resolve_parameters import assert_consistent_resolve_parameters
from cirq.testing.consistent_specified_has_unitary import assert_specifies_has_unitary_if_unitary
from cirq.testing.equivalent_repr_eval import assert_equivalent_repr
from cirq.testing.consistent_controlled_gate_op import (
from cirq.testing.consistent_unitary import assert_unitary_is_consistent
def assert_implements_consistent_protocols(val: Any, *, exponents: Sequence[Any]=(0, 1, -1, 0.25, sympy.Symbol('s')), qubit_count: Optional[int]=None, ignoring_global_phase: bool=False, setup_code: str='import cirq\nimport numpy as np\nimport sympy', global_vals: Optional[Dict[str, Any]]=None, local_vals: Optional[Dict[str, Any]]=None, ignore_decompose_to_default_gateset: bool=False) -> None:
    """Checks that a value is internally consistent and has a good __repr__."""
    global_vals = global_vals or {}
    local_vals = local_vals or {}
    _assert_meets_standards_helper(val, ignoring_global_phase=ignoring_global_phase, setup_code=setup_code, global_vals=global_vals, local_vals=local_vals, ignore_decompose_to_default_gateset=ignore_decompose_to_default_gateset)
    for exponent in exponents:
        p = protocols.pow(val, exponent, None)
        if p is not None:
            _assert_meets_standards_helper(val ** exponent, ignoring_global_phase=ignoring_global_phase, setup_code=setup_code, global_vals=global_vals, local_vals=local_vals, ignore_decompose_to_default_gateset=ignore_decompose_to_default_gateset)