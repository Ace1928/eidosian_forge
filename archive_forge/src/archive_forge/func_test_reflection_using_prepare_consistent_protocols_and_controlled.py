import itertools
import cirq
import cirq_ft
from cirq_ft import infra
import numpy as np
import pytest
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests
@allow_deprecated_cirq_ft_use_in_tests
def test_reflection_using_prepare_consistent_protocols_and_controlled():
    prepare_gate = cirq_ft.StatePreparationAliasSampling.from_lcu_probs([1, 2, 3, 4], probability_epsilon=0.1)
    gate = cirq_ft.ReflectionUsingPrepare(prepare_gate, control_val=None)
    op = gate.on_registers(**infra.get_named_qubits(gate.signature))
    cirq.testing.assert_equivalent_repr(gate, setup_code='import cirq\nimport cirq_ft\nimport numpy as np')
    equals_tester = cirq.testing.EqualsTester()
    equals_tester.add_equality_group(gate.controlled(), gate.controlled(num_controls=1), gate.controlled(control_values=(1,)), op.controlled_by(cirq.q('control')).gate)
    equals_tester.add_equality_group(gate.controlled(control_values=(0,)), gate.controlled(num_controls=1, control_values=(0,)), op.controlled_by(cirq.q('control'), control_values=(0,)).gate)
    with pytest.raises(NotImplementedError, match='Cannot create a controlled version'):
        _ = gate.controlled(num_controls=2)