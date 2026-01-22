import warnings
from typing import cast, Sequence, Union, List, Tuple, Dict, Optional
import numpy as np
import quimb
import quimb.tensor as qtn
import cirq
def tensor_expectation_value(circuit: cirq.Circuit, pauli_string: cirq.PauliString, max_ram_gb=16, tol=1e-06) -> float:
    """Compute an expectation value for an operator and a circuit via tensor
    contraction.

    This will give up if it looks like the computation will take too much RAM.
    """
    circuit_sand = circuit_for_expectation_value(circuit, pauli_string / pauli_string.coefficient)
    qubits = sorted(circuit_sand.all_qubits())
    tensors, qubit_frontier, _ = circuit_to_tensors(circuit=circuit_sand, qubits=qubits)
    end_bras = [qtn.Tensor(data=quimb.up().squeeze(), inds=(f'i{qubit_frontier[q]}_q{q}',), tags={'Q0', 'bra0'}) for q in qubits]
    tn = qtn.TensorNetwork(tensors + end_bras)
    if QUIMB_VERSION[0] < (1, 3):
        warnings.warn(f'quimb version {QUIMB_VERSION[1]} detected. Please use quimb>=1.3 for optimal performance in `tensor_expectation_value`. See https://github.com/quantumlib/Cirq/issues/3263')
    else:
        tn.rank_simplify(inplace=True)
    path_info = tn.contract(get='path-info')
    ram_gb = path_info.largest_intermediate * 128 / 8 / 1024 / 1024 / 1024
    if ram_gb > max_ram_gb:
        raise MemoryError(f'We estimate that this contraction will take too much RAM! {ram_gb} GB')
    e_val = tn.contract(inplace=True)
    if isinstance(e_val, qtn.TensorNetwork):
        e_val = e_val.item()
    assert e_val.imag < tol
    assert cast(complex, pauli_string.coefficient).imag < tol
    return e_val.real * pauli_string.coefficient