from dataclasses import dataclass
from typing import Optional, List, Callable, Dict, Tuple, Set, Any
import networkx as nx
import numpy as np
import pandas as pd
import cirq
import cirq.contrib.routing as ccr
def sample_heavy_set(compilation_result: CompilationResult, heavy_set: List[int], *, repetitions=10000, sampler: cirq.Sampler=cirq.Simulator()) -> float:
    """Run a sampler over the given circuit and compute the percentage of its
       outputs that are in the heavy set.

    Args:
        compilation_result: All the information from the compilation.
        heavy_set: The previously-computed heavy set for the given circuit.
        repetitions: The number of times to sample the circuit.
        sampler: The sampler to run on the given circuit.

    Returns:
        A probability percentage, from 0 to 1, representing how many of the
        output bit-strings were in the heavy set.

    """
    mapping = compilation_result.mapping
    circuit = compilation_result.circuit
    qubits = circuit.all_qubits()
    key = None
    if mapping:
        key = lambda q: mapping.get(q, q)
        qubits = frozenset(mapping.keys())
    sorted_qubits = sorted(qubits, key=key)
    circuit_copy = circuit + [cirq.measure(q) for q in sorted_qubits]
    trial_result = sampler.run(program=circuit_copy, repetitions=repetitions)
    results = process_results(mapping, compilation_result.parity_map, trial_result)
    results = results.agg(lambda meas: cirq.value.big_endian_bits_to_int(meas), axis=1)
    num_in_heavy_set = np.sum(np.in1d(results, heavy_set)).item()
    return num_in_heavy_set / len(results)