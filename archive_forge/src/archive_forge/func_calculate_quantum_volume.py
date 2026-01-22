from dataclasses import dataclass
from typing import Optional, List, Callable, Dict, Tuple, Set, Any
import networkx as nx
import numpy as np
import pandas as pd
import cirq
import cirq.contrib.routing as ccr
def calculate_quantum_volume(*, num_qubits: int, depth: int, num_circuits: int, device_graph: nx.Graph, samplers: List[cirq.Sampler], random_state: 'cirq.RANDOM_STATE_OR_SEED_LIKE'=None, compiler: Optional[Callable[[cirq.Circuit], cirq.Circuit]]=None, repetitions=10000, routing_attempts=30, add_readout_error_correction=False) -> List[QuantumVolumeResult]:
    """Run the quantum volume algorithm.

    This algorithm should compute the same values as Algorithm 1 in
    https://arxiv.org/abs/1811.12926. To summarize, we generate a random model
    circuit, compute its heavy set, then transpile an implementation onto our
    architecture. This implementation is run a series of times and if the
    percentage of outputs that are in the heavy set is greater than 2/3, we
    consider the quantum volume test passed for that size.

    Args:
        num_qubits: The number of qubits for the circuit.
        depth: The number of gate layers to generate.
        num_circuits: The number of random circuits to run.
        random_state: Random state or random state seed.
        device_graph: A graph whose nodes are qubits and edges represent two
            qubit interactions to run the compiled circuit on.
        samplers: The samplers to run the algorithm on.
        compiler: An optional function to compiler the model circuit's
            gates down to the target devices gate set and the optimize it.
        repetitions: The number of bitstrings to sample per circuit.
        routing_attempts: The number of times to route each model circuit onto
            the device. Each attempt will be graded using an ideal simulator
            and the best one will be used.
        add_readout_error_correction: If true, add some parity bits that will
            later be used to detect readout error. WARNING: This makes the
            simulator run extremely slowly for any width/depth of 4 or more,
            because it doubles the circuit size. In reality, the simulator
            shouldn't need to use this larger circuit for the majority of
            operations, since they only come into play at the end.

    Returns: A list of QuantumVolumeResults that contains all of the information
        for running the algorithm and its results.

    """
    circuits = prepare_circuits(num_qubits=num_qubits, depth=depth, num_circuits=num_circuits, random_state=random_state)
    return execute_circuits(circuits=circuits, device_graph=device_graph, compiler=compiler, samplers=samplers, repetitions=repetitions, routing_attempts=routing_attempts, add_readout_error_correction=add_readout_error_correction)