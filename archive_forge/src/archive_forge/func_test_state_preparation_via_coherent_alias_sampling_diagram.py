import cirq
import cirq_ft
import numpy as np
import pytest
from cirq_ft.algos.generic_select_test import get_1d_Ising_lcu_coeffs
from cirq_ft.infra.jupyter_tools import execute_notebook
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests
@allow_deprecated_cirq_ft_use_in_tests
def test_state_preparation_via_coherent_alias_sampling_diagram():
    data = np.asarray(range(1, 5)) / np.sum(range(1, 5))
    gate = cirq_ft.StatePreparationAliasSampling.from_lcu_probs(lcu_probabilities=data.tolist(), probability_epsilon=0.05)
    g = cirq_ft.testing.GateHelper(gate)
    qubit_order = g.operation.qubits
    circuit = cirq.Circuit(cirq.decompose_once(g.operation))
    cirq.testing.assert_has_diagram(circuit, '\nselection0: ────────UNIFORM(4)───In───────────────────×(y)───\n                    │            │                    │\nselection1: ────────target───────In───────────────────×(y)───\n                                 │                    │\nsigma_mu0: ─────────H────────────┼────────In(y)───────┼──────\n                                 │        │           │\nsigma_mu1: ─────────H────────────┼────────In(y)───────┼──────\n                                 │        │           │\nsigma_mu2: ─────────H────────────┼────────In(y)───────┼──────\n                                 │        │           │\nalt0: ───────────────────────────QROM_0───┼───────────×(x)───\n                                 │        │           │\nalt1: ───────────────────────────QROM_0───┼───────────×(x)───\n                                 │        │           │\nkeep0: ──────────────────────────QROM_1───In(x)───────┼──────\n                                 │        │           │\nkeep1: ──────────────────────────QROM_1───In(x)───────┼──────\n                                 │        │           │\nkeep2: ──────────────────────────QROM_1───In(x)───────┼──────\n                                          │           │\nless_than_equal: ─────────────────────────+(x <= y)───@──────\n', qubit_order=qubit_order)