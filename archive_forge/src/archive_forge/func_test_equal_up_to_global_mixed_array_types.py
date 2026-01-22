import numpy as np
import cirq
def test_equal_up_to_global_mixed_array_types():
    a = [1j, 1, -1j, -1]
    b = [-1, 1j, 1, -1j]
    c = [-1, 1, -1, 1]
    assert cirq.equal_up_to_global_phase(a, tuple(b))
    assert not cirq.equal_up_to_global_phase(a, tuple(c))
    c_types = [np.complex64, np.complex128]
    if hasattr(np, 'complex256'):
        c_types.append(np.complex256)
    for c_type in c_types:
        assert cirq.equal_up_to_global_phase(np.asarray(a, dtype=c_type), tuple(b))
        assert not cirq.equal_up_to_global_phase(np.asarray(a, dtype=c_type), tuple(c))
        assert cirq.equal_up_to_global_phase(np.asarray(a, dtype=c_type), b)
        assert not cirq.equal_up_to_global_phase(np.asarray(a, dtype=c_type), c)
    assert not cirq.equal_up_to_global_phase([1j], 1j)
    assert not cirq.equal_up_to_global_phase(np.asarray([1], dtype=np.complex128), np.exp(1j))
    assert not cirq.equal_up_to_global_phase([1j, 1j], [1j, '1j'])
    assert not cirq.equal_up_to_global_phase([1j], 'Non-numeric iterable')
    assert not cirq.equal_up_to_global_phase([], [[]], atol=0.0)