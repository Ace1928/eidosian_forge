import numpy as np
import pytest
import cirq
from cirq.protocols.apply_unitary_protocol import _incorporate_result_into_target
def test_apply_unitary_presence_absence():
    m = np.diag([1, -1])

    class NoUnitaryEffect:
        pass

    class HasUnitary:

        def _unitary_(self) -> np.ndarray:
            return m

    class HasApplyReturnsNotImplemented:

        def _apply_unitary_(self, args: cirq.ApplyUnitaryArgs):
            return NotImplemented

    class HasApplyReturnsNotImplementedButHasUnitary:

        def _apply_unitary_(self, args: cirq.ApplyUnitaryArgs):
            return NotImplemented

        def _unitary_(self) -> np.ndarray:
            return m

    class HasApplyOutputInBuffer:

        def _apply_unitary_(self, args: cirq.ApplyUnitaryArgs) -> np.ndarray:
            zero = args.subspace_index(0)
            one = args.subspace_index(1)
            args.available_buffer[zero] = args.target_tensor[zero]
            args.available_buffer[one] = -args.target_tensor[one]
            return args.available_buffer

    class HasApplyMutateInline:

        def _apply_unitary_(self, args: cirq.ApplyUnitaryArgs) -> np.ndarray:
            one = args.subspace_index(1)
            args.target_tensor[one] *= -1
            return args.target_tensor
    fails = [NoUnitaryEffect(), HasApplyReturnsNotImplemented()]
    passes = [HasUnitary(), HasApplyReturnsNotImplementedButHasUnitary(), HasApplyOutputInBuffer(), HasApplyMutateInline()]

    def make_input():
        return np.ones((2, 2))

    def assert_works(val):
        expected_outputs = [np.array([1, 1, -1, -1]).reshape((2, 2)), np.array([1, -1, 1, -1]).reshape((2, 2))]
        for axis in range(2):
            result = cirq.apply_unitary(val, cirq.ApplyUnitaryArgs(make_input(), buf, [axis]))
            np.testing.assert_allclose(result, expected_outputs[axis])
    buf = np.empty(shape=(2, 2), dtype=np.complex128)
    for f in fails:
        with pytest.raises(TypeError, match='failed to satisfy'):
            _ = cirq.apply_unitary(f, cirq.ApplyUnitaryArgs(make_input(), buf, [0]))
        assert cirq.apply_unitary(f, cirq.ApplyUnitaryArgs(make_input(), buf, [0]), default=None) is None
        assert cirq.apply_unitary(f, cirq.ApplyUnitaryArgs(make_input(), buf, [0]), default=NotImplemented) is NotImplemented
        assert cirq.apply_unitary(f, cirq.ApplyUnitaryArgs(make_input(), buf, [0]), default=1) == 1
    for s in passes:
        assert_works(s)
        assert cirq.apply_unitary(s, cirq.ApplyUnitaryArgs(make_input(), buf, [0]), default=None) is not None