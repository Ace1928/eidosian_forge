import cirq
import cirq_ft
import numpy as np
import pytest
from cirq_ft import infra
from cirq_ft.algos.generic_select_test import get_1d_Ising_hamiltonian
from cirq_ft.algos.reflection_using_prepare_test import greedily_allocate_ancilla, keep
from cirq_ft.infra.jupyter_tools import execute_notebook
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests
def get_walk_operator_for_1d_Ising_model(num_sites: int, eps: float) -> cirq_ft.QubitizationWalkOperator:
    ham = get_1d_Ising_hamiltonian(cirq.LineQubit.range(num_sites))
    return walk_operator_for_pauli_hamiltonian(ham, eps)