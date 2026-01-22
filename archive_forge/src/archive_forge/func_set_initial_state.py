import warnings
from typing import Any, List, Optional, Sequence, Union
import numpy as np
from numpy.random.mtrand import RandomState
from pyquil.paulis import PauliTerm, PauliSum
from pyquil.pyqvm import AbstractQuantumSimulator
from pyquil.quilbase import Gate
from pyquil.simulation.matrices import P0, P1, KRAUS_OPS, QUANTUM_GATES
from pyquil.simulation.tools import lifted_gate_matrix, lifted_gate, all_bitstrings
def set_initial_state(self, state_matrix: np.ndarray) -> 'ReferenceDensitySimulator':
    """
        This method is the correct way (TM) to update the initial state matrix that is
        initialized every time reset() is called. The default initial state of
        ReferenceDensitySimulator is ``|000...00>``.

        Note that the current state matrix, i.e. ``self.density`` is not affected by this
        method; you must change it directly or else call reset() after calling this method.

        To restore default state initialization behavior of ReferenceDensitySimulator pass in
        a ``state_matrix`` equal to the default initial state on `n_qubits` (i.e. ``|000...00>``)
        and then call ``reset()``. We have provided a helper function ``n_qubit_zero_state``
        in the ``_reference.py`` module to simplify this step.

        :param state_matrix: numpy.ndarray or None.
        :return: ``self`` to support method chaining.
        """
    rows, cols = state_matrix.shape
    if rows != cols:
        raise ValueError('The state matrix is not square.')
    if self.n_qubits != int(np.log2(rows)):
        raise ValueError('The state matrix is not defined on the same numbers of qubits as the QVM.')
    if _is_valid_quantum_state(state_matrix):
        self.initial_density = state_matrix
    else:
        raise ValueError('The state matrix is not valid. It must be Hermitian, trace one, and have non-negative eigenvalues.')
    return self