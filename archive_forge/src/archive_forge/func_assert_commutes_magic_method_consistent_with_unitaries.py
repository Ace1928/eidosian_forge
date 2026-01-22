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
def assert_commutes_magic_method_consistent_with_unitaries(*vals: Sequence[Any], atol: Union[int, float]=1e-08) -> None:
    if any((isinstance(val, ops.Operation) for val in vals)):
        raise TypeError('`_commutes_` need not be consistent with unitaries for `Operation`.')
    unitaries = [protocols.unitary(val, None) for val in vals]
    pairs = itertools.permutations(zip(vals, unitaries), 2)
    for (left_val, left_unitary), (right_val, right_unitary) in pairs:
        if left_unitary is None or right_unitary is None:
            continue
        commutes = protocols.commutes(left_val, right_val, atol=atol, default=None)
        if commutes is None:
            continue
        assert commutes == protocols.commutes(left_unitary, right_unitary)