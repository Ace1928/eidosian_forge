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
def test_benchmark_2q_xeb_subsample_depths(circuits_cycle_depths_sampled_df):
    circuits, _, sampled_df = circuits_cycle_depths_sampled_df
    cycle_depths = [10, 20]
    fid_df = benchmark_2q_xeb_fidelities(sampled_df, circuits, cycle_depths)
    assert len(fid_df) == len(cycle_depths)
    assert sorted(fid_df['cycle_depth'].unique()) == cycle_depths
    cycle_depths = [11, 21]
    with pytest.raises(ValueError):
        _ = benchmark_2q_xeb_fidelities(sampled_df, circuits, cycle_depths)
    cycle_depths = [10, 100000]
    with pytest.raises(ValueError):
        _ = benchmark_2q_xeb_fidelities(sampled_df, circuits, cycle_depths)
    cycle_depths = []
    with pytest.raises(ValueError):
        _ = benchmark_2q_xeb_fidelities(sampled_df, circuits, cycle_depths)