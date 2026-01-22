from typing import Any, List, Optional, Sequence, Tuple, Union, cast
import numpy as np
from numpy.random.mtrand import RandomState
from pyquil.paulis import PauliTerm, PauliSum
from pyquil.pyqvm import AbstractQuantumSimulator
from pyquil.quilbase import Gate
from pyquil.simulation.matrices import QUANTUM_GATES
from pyquil.simulation.tools import all_bitstrings
def targeted_tensordot(gate: np.ndarray, wf: np.ndarray, wf_target_inds: Sequence[int]) -> np.ndarray:
    """Left-multiplies the given axes of the wf tensor by the given gate matrix.

    Compare with :py:func:`targeted_einsum`. The semantics of these two functions should be
    identical, except this uses ``np.tensordot`` instead of ``np.einsum``.

    :param gate: What to left-multiply the target tensor by.
    :param wf: A tensor to carefully broadcast a left-multiply over.
    :param wf_target_inds: Which axes of the target are being operated on.
    :returns: The output tensor.
    """
    gate_n_qubits = gate.ndim // 2
    n_qubits = wf.ndim
    gate_inds = np.arange(gate_n_qubits, 2 * gate_n_qubits)
    assert len(wf_target_inds) == len(gate_inds), (wf_target_inds, gate_inds)
    wf = np.tensordot(gate, wf, (gate_inds, wf_target_inds))
    axes_ordering = list(range(gate_n_qubits, n_qubits))
    where_td_put_them = np.arange(gate_n_qubits)
    sorty = np.argsort(wf_target_inds)
    where_td_put_them = where_td_put_them[sorty]
    sorted_targets = np.asarray(wf_target_inds)[sorty]
    for target_ind, from_ind in zip(sorted_targets, where_td_put_them):
        axes_ordering.insert(target_ind, from_ind)
    return wf.transpose(axes_ordering)