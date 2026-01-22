import dataclasses
import itertools
from typing import Any, cast, Iterator, List, Optional, Sequence, Tuple, TYPE_CHECKING
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from cirq import circuits, ops, protocols
def two_qubit_randomized_benchmarking(sampler: 'cirq.Sampler', first_qubit: 'cirq.Qid', second_qubit: 'cirq.Qid', *, num_clifford_range: Sequence[int]=range(5, 50, 5), num_circuits: int=20, repetitions: int=1000) -> RandomizedBenchMarkResult:
    """Clifford-based randomized benchmarking (RB) of two qubits.

    A total of num_circuits random circuits are generated, each of which
    contains a fixed number of two-qubit Clifford gates plus one additional
    Clifford that inverts the whole sequence and a measurement in the
    z-basis. Each circuit is repeated a number of times and the average
    |00> state population is determined from the measurement outcomes of all
    of the circuits.

    The above process is done for different circuit lengths specified by the
    integers in num_clifford_range. For example, an integer 10 means the
    random circuits will contain 10 Clifford gates each plus one inverting
    Clifford. The user may use the result to extract an average gate fidelity,
    by analyzing the change in the average |00> state population at different
    circuit lengths. For actual experiments, one should choose
    num_clifford_range such that a clear exponential decay is observed in the
    results.

    The two-qubit Cliffords here are decomposed into CZ gates plus single-qubit
    x and y rotations. See Barends et al., Nature 508, 500 for details.

    Args:
        sampler: The quantum engine or simulator to run the circuits.
        first_qubit: The first qubit under test.
        second_qubit: The second qubit under test.
        num_clifford_range: The different numbers of Cliffords in the RB study.
        num_circuits: The number of random circuits generated for each
            number of Cliffords.
        repetitions: The number of repetitions of each circuit.

    Returns:
        A RandomizedBenchMarkResult object that stores and plots the result.
    """
    cliffords = _single_qubit_cliffords()
    cfd_matrices = _two_qubit_clifford_matrices(first_qubit, second_qubit, cliffords)
    gnd_probs = []
    for num_cfds in num_clifford_range:
        gnd_probs_l = []
        for _ in range(num_circuits):
            circuit = _random_two_q_clifford(first_qubit, second_qubit, num_cfds, cfd_matrices, cliffords)
            circuit.append(ops.measure(first_qubit, second_qubit, key='z'))
            results = sampler.run(circuit, repetitions=repetitions)
            gnds = [not r[0] and (not r[1]) for r in results.measurements['z']]
            gnd_probs_l.append(np.mean(gnds))
        gnd_probs.append(float(np.mean(gnd_probs_l)))
    return RandomizedBenchMarkResult(num_clifford_range, gnd_probs)