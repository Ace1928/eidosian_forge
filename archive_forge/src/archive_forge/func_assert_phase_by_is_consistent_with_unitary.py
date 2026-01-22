from typing import Any
import numpy as np
import sympy
from cirq import protocols, linalg
from cirq.testing import lin_alg_utils
def assert_phase_by_is_consistent_with_unitary(val: Any):
    """Uses `val._unitary_` to check `val._phase_by_`'s behavior."""
    original = protocols.unitary(val, None)
    if original is None:
        return
    qid_shape = protocols.qid_shape(val, default=(2,) * (len(original).bit_length() - 1))
    original = original.reshape(qid_shape * 2)
    for t in [0.125, -0.25, 1, sympy.Symbol('a'), sympy.Symbol('a') + 1]:
        p = 1j ** (t * 4)
        p = protocols.resolve_parameters(p, {'a': -0.125})
        for i in range(len(qid_shape)):
            phased = protocols.phase_by(val, t, i, default=None)
            if phased is None:
                continue
            phased = protocols.resolve_parameters(phased, {'a': -0.125})
            actual = protocols.unitary(phased).reshape(qid_shape * 2)
            expected = np.array(original)
            s = linalg.slice_for_qubits_equal_to([i], 1)
            expected[s] *= p
            s = linalg.slice_for_qubits_equal_to([len(qid_shape) + i], 1)
            expected[s] *= np.conj(p)
            lin_alg_utils.assert_allclose_up_to_global_phase(actual, expected, atol=1e-08, err_msg=f'Phased unitary was incorrect for index #{i}')