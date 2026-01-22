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
def test_fit_exponential_decays_negative_fids():
    rs = np.random.RandomState(999)
    cycle_depths = np.arange(3, 100, 11)
    fidelities = 0.5 * 0.5 ** cycle_depths + rs.normal(0, 0.2) - 0.5
    assert np.sum(fidelities > 0) <= 1, 'they go negative'
    a, layer_fid, a_std, layer_fid_std = _fit_exponential_decay(cycle_depths, fidelities)
    assert a == 0
    assert layer_fid == 0
    assert a_std == np.inf
    assert layer_fid_std == np.inf