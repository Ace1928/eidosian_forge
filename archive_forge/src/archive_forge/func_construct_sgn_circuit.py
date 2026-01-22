import json
from os import path
from typing import Sequence, Callable
import numpy as np
import pennylane as qml
from pennylane.transforms import transform
def construct_sgn_circuit(hamiltonian, tape, mus, times, phis, controls):
    """
    Takes a tape with state prep and ansatz and constructs the individual tapes
    approximating/estimating the individual terms of your decomposition

    Args:
      hamiltonian (qml.Hamiltonian): The pennylane Hamiltonian to be decomposed
      tape (qml.QuantumTape: Tape containing the circuit to be expanded into the new circuits
      mus (List[float]): The average between the two eigenvalues (E_1+E-2)/2
      times (List[float]): The time for this term group to be evaluated/evolved at
      phis (List[float]): Optimal phi values for the QSP part associated with the respective
        delta and J
      controls (List[control1, control2]): The two additional controls to implement the
          Hadamard test and the quantum signal processing part on

    Returns:
      tapes (List[qml.tape]): Expanded tapes from the original tape that measures the terms
        via the approximate sgn decomposition
    """
    coeffs = hamiltonian.data
    tapes = []
    for mu, time in zip(mus, times):
        added_operations = []
        added_operations.append(qml.Hadamard(controls[0]))
        for i, phi in enumerate(phis):
            added_operations.append(qml.CRX(phi, wires=controls))
            if i == len(phis) - 1:
                added_operations.append(qml.CRY(np.pi, wires=controls))
            else:
                for ops in evolve_under(hamiltonian.ops, coeffs, 2 * time, controls):
                    added_operations.extend(ops)
                added_operations.append(qml.CRZ(-2 * mu * time, wires=controls))
        added_operations.append(qml.Hadamard(controls[0]))
        operations = tape.operations + added_operations
        if tape.measurements[0].return_type == qml.measurements.Expectation:
            measurements = [qml.expval(-1 * qml.Z(controls[0]))]
        else:
            measurements = [qml.var(qml.Z(controls[0]))]
        new_tape = qml.tape.QuantumScript(operations, measurements, shots=tape.shots)
        tapes.append(new_tape)
    return tapes