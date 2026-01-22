import dataclasses
import itertools
from typing import Any, cast, Iterator, List, Optional, Sequence, Tuple, TYPE_CHECKING
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from cirq import circuits, ops, protocols
def single_qubit_randomized_benchmarking(sampler: 'cirq.Sampler', qubit: 'cirq.Qid', use_xy_basis: bool=True, *, num_clifford_range: Sequence[int]=range(10, 100, 10), num_circuits: int=20, repetitions: int=1000) -> RandomizedBenchMarkResult:
    """Clifford-based randomized benchmarking (RB) of a single qubit.

    A total of num_circuits random circuits are generated, each of which
    contains a fixed number of single-qubit Clifford gates plus one
    additional Clifford that inverts the whole sequence and a measurement in
    the z-basis. Each circuit is repeated a number of times and the average
    |0> state population is determined from the measurement outcomes of all
    of the circuits.

    The above process is done for different circuit lengths specified by the
    integers in num_clifford_range. For example, an integer 10 means the
    random circuits will contain 10 Clifford gates each plus one inverting
    Clifford. The user may use the result to extract an average gate fidelity,
    by analyzing the change in the average |0> state population at different
    circuit lengths. For actual experiments, one should choose
    num_clifford_range such that a clear exponential decay is observed in the
    results.

    See Barends et al., Nature 508, 500 for details.

    Args:
        sampler: The quantum engine or simulator to run the circuits.
        qubit: The qubit under test.
        use_xy_basis: Determines if the Clifford gates are built with x and y
            rotations (True) or x and z rotations (False).
        num_clifford_range: The different numbers of Cliffords in the RB study.
        num_circuits: The number of random circuits generated for each
            number of Cliffords.
        repetitions: The number of repetitions of each circuit.

    Returns:
        A RandomizedBenchMarkResult object that stores and plots the result.
    """
    cliffords = _single_qubit_cliffords()
    c1 = cliffords.c1_in_xy if use_xy_basis else cliffords.c1_in_xz
    cfd_mats = np.array([_gate_seq_to_mats(gates) for gates in c1])
    gnd_probs = []
    for num_cfds in num_clifford_range:
        excited_probs_l = []
        for _ in range(num_circuits):
            circuit = _random_single_q_clifford(qubit, num_cfds, c1, cfd_mats)
            circuit.append(ops.measure(qubit, key='z'))
            results = sampler.run(circuit, repetitions=repetitions)
            excited_probs_l.append(np.mean(results.measurements['z']))
        gnd_probs.append(1.0 - np.mean(excited_probs_l))
    return RandomizedBenchMarkResult(num_clifford_range, gnd_probs)