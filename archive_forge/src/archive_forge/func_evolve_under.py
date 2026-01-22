import json
from os import path
from typing import Sequence, Callable
import numpy as np
import pennylane as qml
from pennylane.transforms import transform
def evolve_under(ops, coeffs, time, controls):
    """
    Evolves under the given Hamiltonian deconstructed into its Pauli words

    Args:
        ops (List[Observables]): List of Pauli words that comprise the Hamiltonian
        coeffs (List[int]): List of the respective coefficients of the Pauliwords of the Hamiltonian
        time (float): At what time to evaluate these Pauliwords
    """
    ops_temp = []
    for op, coeff in zip(ops, coeffs):
        pauli_word = qml.pauli.pauli_word_to_string(op)
        ops_temp.append(controlled_pauli_evolution(coeff * time, wires=op.wires, pauli_word=pauli_word, controls=controls))
    return ops_temp