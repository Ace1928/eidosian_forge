import warnings
from typing import cast, Sequence, Union, List, Tuple, Dict, Optional
import numpy as np
import quimb
import quimb.tensor as qtn
import cirq
def tensor_unitary(circuit: cirq.Circuit, qubits: Optional[Sequence[cirq.Qid]]=None) -> np.ndarray:
    """Given a circuit contract a tensor network into a dense unitary
    of the circuit."""
    if qubits is None:
        qubits = sorted(circuit.all_qubits())
    tensors, qubit_frontier, _ = circuit_to_tensors(circuit=circuit, qubits=qubits, initial_state=None)
    tn = qtn.TensorNetwork(tensors)
    i_inds = tuple((f'i0_q{q}' for q in qubits))
    f_inds = tuple((f'i{qubit_frontier[q]}_q{q}' for q in qubits))
    tn.contract(inplace=True)
    return tn.to_dense(f_inds, i_inds)