import itertools
import multiprocessing
from typing import Iterable
import networkx as nx
import numpy as np
import pandas as pd
import pytest
import cirq
import cirq.experiments.random_quantum_circuit_generation as rqcg
from cirq.experiments.xeb_fitting import (
from cirq.experiments.xeb_sampling import sample_2q_xeb_circuits
@pytest.fixture(scope='module')
def circuits_cycle_depths_sampled_df():
    q0, q1 = cirq.LineQubit.range(2)
    circuits = [rqcg.random_rotations_between_two_qubit_circuit(q0, q1, depth=50, two_qubit_op_factory=lambda a, b, _: cirq.SQRT_ISWAP(a, b), seed=52) for _ in range(2)]
    cycle_depths = np.arange(10, 40 + 1, 10)
    sampled_df = sample_2q_xeb_circuits(sampler=cirq.Simulator(seed=53), circuits=circuits, cycle_depths=cycle_depths)
    return (circuits, cycle_depths, sampled_df)