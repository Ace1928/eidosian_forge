from typing import List, TYPE_CHECKING
import functools
import numpy as np
from cirq import ops, protocols, qis, sim
def decompose_clifford_tableau_to_operations(qubits: List['cirq.Qid'], clifford_tableau: qis.CliffordTableau) -> List[ops.Operation]:
    """Decompose an n-qubit Clifford Tableau into a list of one/two qubit operations.

    The implementation is based on Theorem 8 in [1].
    [1] S. Aaronson, D. Gottesman, *Improved Simulation of Stabilizer Circuits*,
        Phys. Rev. A 70, 052328 (2004). https://arxiv.org/abs/quant-ph/0406196

    Args:
        qubits: The list of qubits being operated on.
        clifford_tableau: The Clifford Tableau for decomposition.

    Returns:
        A list of operations reconstructs the same Clifford tableau.

    Raises:
        ValueError: The length of input qubit mismatch with the size of tableau.
    """
    if len(qubits) != clifford_tableau.n:
        raise ValueError('The number of qubits must be the same as the number of Clifford Tableau.')
    assert clifford_tableau._validate(), 'The provided clifford_tableau must satisfy the symplectic property.'
    t: qis.CliffordTableau = clifford_tableau.copy()
    operations: List[ops.Operation] = []
    args = sim.CliffordTableauSimulationState(tableau=t, qubits=qubits, prng=np.random.RandomState())
    _X_with_ops = functools.partial(_X, args=args, operations=operations, qubits=qubits)
    _Z_with_ops = functools.partial(_Z, args=args, operations=operations, qubits=qubits)
    _H_with_ops = functools.partial(_H, args=args, operations=operations, qubits=qubits)
    _S_with_ops = functools.partial(_Sdg, args=args, operations=operations, qubits=qubits)
    _CNOT_with_ops = functools.partial(_CNOT, args=args, operations=operations, qubits=qubits)
    _SWAP_with_ops = functools.partial(_SWAP, args=args, operations=operations, qubits=qubits)
    for i in range(t.n):
        if not t.xs[i, i] and t.zs[i, i]:
            _H_with_ops(i)
        if not t.xs[i, i]:
            for j in range(i + 1, t.n):
                if t.xs[i, j]:
                    _SWAP_with_ops(i, j)
                    break
        if not t.xs[i, i]:
            for j in range(i + 1, t.n):
                if t.zs[i, j]:
                    _H_with_ops(j)
                    _SWAP_with_ops(i, j)
                    break
        _ = [_CNOT_with_ops(i, j) for j in range(i + 1, t.n) if t.xs[i, j]]
        if np.any(t.zs[i, i:]):
            if not t.zs[i, i]:
                _S_with_ops(i)
            _ = [_CNOT_with_ops(j, i) for j in range(i + 1, t.n) if t.zs[i, j]]
            _S_with_ops(i)
        _ = [_CNOT_with_ops(j, i) for j in range(i + 1, t.n) if t.zs[i + t.n, j]]
        if np.any(t.xs[i + t.n, i:]):
            _H_with_ops(i)
            _ = [_CNOT_with_ops(i, j) for j in range(i + 1, t.n) if t.xs[i + t.n, j]]
            if t.zs[i + t.n, i]:
                _S_with_ops(i)
            _H_with_ops(i)
    _ = [_Z_with_ops(i) for i, p in enumerate(t.rs[:t.n]) if p]
    _ = [_X_with_ops(i) for i, p in enumerate(t.rs[t.n:]) if p]
    return operations[::-1]