from functools import lru_cache
from typing import Sequence, Dict, Union, Tuple, List, Optional
import numpy as np
import quimb
import quimb.tensor as qtn
import cirq
def circuit_to_density_matrix_tensors(circuit: cirq.Circuit, qubits: Optional[Sequence[cirq.Qid]]=None) -> Tuple[List[qtn.Tensor], Dict['cirq.Qid', int], Dict[Tuple[str, str], Tuple[float, float]]]:
    """Given a circuit with mixtures or channels, construct a tensor network
    representation of the density matrix.

    This assumes you start in the |0..0><0..0| state. Indices are named
    "nf{i}_q{x}" and "nb{i}_q{x}" where i is a time index and x is a
    qubit index. nf- and nb- refer to the "forwards" and "backwards"
    copies of the circuit. Kraus indices are named "k{j}" where j is an
    independent "kraus" internal index which you probably never need to access.

    Args:
        circuit: The circuit containing operations that support the
            cirq.unitary() or cirq.kraus() protocols.
        qubits: The qubits in the circuit. The `positions` return argument
            will position qubits according to their index in this list.

    Returns:
        tensors: A list of Quimb Tensor objects
        qubit_frontier: A mapping from qubit to time index at the end of
            the circuit. This can be used to deduce the names of the free
            tensor indices.
        positions: A positions dictionary suitable for passing to tn.graph()'s
            `fix` argument to draw the resulting tensor network similar to a
            quantum circuit.

    Raises:
        ValueError: If an op is encountered that cannot be converted.
    """
    if qubits is None:
        qubits = sorted(circuit.all_qubits())
    qubits = tuple(qubits)
    qubit_frontier: Dict[cirq.Qid, int] = {q: 0 for q in qubits}
    kraus_frontier = 0
    positions: Dict[Tuple[str, str], Tuple[float, float]] = {}
    tensors: List[qtn.Tensor] = []
    x_scale = 2
    y_scale = 3
    x_nudge = 0.3
    n_qubits = len(qubits)
    yb_offset = (n_qubits + 0.5) * y_scale

    def _positions(_mi, _these_qubits):
        return _add_to_positions(positions, _mi, _these_qubits, all_qubits=qubits, x_scale=x_scale, y_scale=y_scale, x_nudge=x_nudge, yb_offset=yb_offset)
    for q in qubits:
        tensors += [qtn.Tensor(data=quimb.up().squeeze(), inds=(f'nf0_q{q}',), tags={'Q0', 'i0f', _qpos_tag(q)}), qtn.Tensor(data=quimb.up().squeeze(), inds=(f'nb0_q{q}',), tags={'Q0', 'i0b', _qpos_tag(q)})]
        _positions(0, q)
    for mi, moment in enumerate(circuit.moments):
        for op in moment.operations:
            start_inds_f = [f'nf{qubit_frontier[q]}_q{q}' for q in op.qubits]
            start_inds_b = [f'nb{qubit_frontier[q]}_q{q}' for q in op.qubits]
            for q in op.qubits:
                qubit_frontier[q] += 1
            end_inds_f = [f'nf{qubit_frontier[q]}_q{q}' for q in op.qubits]
            end_inds_b = [f'nb{qubit_frontier[q]}_q{q}' for q in op.qubits]
            if cirq.has_unitary(op):
                U = cirq.unitary(op).reshape((2,) * 2 * len(op.qubits)).astype(np.complex128)
                tensors.append(qtn.Tensor(data=U, inds=end_inds_f + start_inds_f, tags={f'Q{len(op.qubits)}', f'i{mi + 1}f', _qpos_tag(op.qubits)}))
                tensors.append(qtn.Tensor(data=np.conj(U), inds=end_inds_b + start_inds_b, tags={f'Q{len(op.qubits)}', f'i{mi + 1}b', _qpos_tag(op.qubits)}))
            elif cirq.has_kraus(op):
                K = np.asarray(cirq.kraus(op), dtype=np.complex128)
                kraus_inds = [f'k{kraus_frontier}']
                tensors.append(qtn.Tensor(data=K, inds=kraus_inds + end_inds_f + start_inds_f, tags={f'kQ{len(op.qubits)}', f'i{mi + 1}f', _qpos_tag(op.qubits)}))
                tensors.append(qtn.Tensor(data=np.conj(K), inds=kraus_inds + end_inds_b + start_inds_b, tags={f'kQ{len(op.qubits)}', f'i{mi + 1}b', _qpos_tag(op.qubits)}))
                kraus_frontier += 1
            else:
                raise ValueError(repr(op))
            _positions(mi + 1, op.qubits)
    return (tensors, qubit_frontier, positions)