import glob
import itertools
from typing import Iterable
import networkx as nx
import numpy as np
import pandas as pd
import pytest
import cirq
import cirq.experiments.random_quantum_circuit_generation as rqcg
from cirq.experiments.xeb_sampling import sample_2q_xeb_circuits
def test_sample_2q_parallel_xeb_circuits_error_bad_qubits():
    circuits = rqcg.generate_library_of_2q_circuits(n_library_circuits=5, two_qubit_gate=cirq.ISWAP ** 0.5, max_cycle_depth=10, q0=cirq.GridQubit(0, 0), q1=cirq.GridQubit(1, 1))
    cycle_depths = [10]
    graph = _gridqubits_to_graph_device(cirq.GridQubit.rect(3, 2))
    combs = rqcg.get_random_combinations_for_device(n_library_circuits=len(circuits), n_combinations=5, device_graph=graph, random_state=10)
    with pytest.raises(ValueError, match='.*each operating on LineQubit\\(0\\) and LineQubit\\(1\\)'):
        _ = sample_2q_xeb_circuits(sampler=cirq.Simulator(), circuits=circuits, cycle_depths=cycle_depths, combinations_by_layer=combs)