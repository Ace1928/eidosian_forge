import numpy as np
import pytest
import cirq
from cirq.protocols.apply_unitary_protocol import _incorporate_result_into_target
def test_apply_unitary_args_tensor_manipulation():

    class ModifyTargetTensor:

        def _apply_unitary_(self, args):
            zo = args.subspace_index(1)
            oz = args.subspace_index(2)
            args.available_buffer[zo] = args.target_tensor[zo]
            args.target_tensor[zo] = args.target_tensor[oz]
            args.target_tensor[oz] = args.available_buffer[zo]
            args.target_tensor[...] *= 1j
            args.available_buffer[...] = 99
            return args.target_tensor

    class TransposeTargetTensor:

        def _apply_unitary_(self, args):
            indices = list(range(len(args.target_tensor.shape)))
            indices[args.axes[0]], indices[args.axes[1]] = (indices[args.axes[1]], indices[args.axes[0]])
            target = args.target_tensor.transpose(*indices)
            target[...] *= 1j
            args.available_buffer[...] = 99
            return target

    class ReshapeTargetTensor:

        def _apply_unitary_(self, args):
            zz = args.subspace_index(0)
            zo = args.subspace_index(1)
            oz = args.subspace_index(2)
            oo = args.subspace_index(3)
            args.available_buffer[zz] = args.target_tensor[zz]
            args.available_buffer[zo] = args.target_tensor[zo]
            args.available_buffer[oz] = args.target_tensor[oz]
            args.available_buffer[oo] = args.target_tensor[oo]
            target = args.target_tensor.transpose(*range(1, len(args.target_tensor.shape)), 0).reshape(args.target_tensor.shape)
            target[zz] = args.available_buffer[zz]
            target[zo] = args.available_buffer[oz]
            target[oz] = args.available_buffer[zo]
            target[oo] = args.available_buffer[oo]
            target[...] *= 1j
            args.available_buffer[...] = 99
            return target

    class ModifyAvailableBuffer:

        def _apply_unitary_(self, args):
            zz = args.subspace_index(0)
            zo = args.subspace_index(1)
            oz = args.subspace_index(2)
            oo = args.subspace_index(3)
            args.available_buffer[zz] = args.target_tensor[zz]
            args.available_buffer[zo] = args.target_tensor[oz]
            args.available_buffer[oz] = args.target_tensor[zo]
            args.available_buffer[oo] = args.target_tensor[oo]
            args.available_buffer[...] *= 1j
            args.target_tensor[...] = 99
            return args.available_buffer

    class TransposeAvailableBuffer:

        def _apply_unitary_(self, args):
            indices = list(range(len(args.target_tensor.shape)))
            indices[args.axes[0]], indices[args.axes[1]] = (indices[args.axes[1]], indices[args.axes[0]])
            output = args.available_buffer.transpose(*indices)
            args.available_buffer[...] = args.target_tensor
            output *= 1j
            args.target_tensor[...] = 99
            return output

    class ReshapeAvailableBuffer:

        def _apply_unitary_(self, args):
            zz = args.subspace_index(0)
            zo = args.subspace_index(1)
            oz = args.subspace_index(2)
            oo = args.subspace_index(3)
            output = args.available_buffer.transpose(*range(1, len(args.available_buffer.shape)), 0).reshape(args.available_buffer.shape)
            output[zz] = args.target_tensor[zz]
            output[zo] = args.target_tensor[oz]
            output[oz] = args.target_tensor[zo]
            output[oo] = args.target_tensor[oo]
            output[...] *= 1j
            args.target_tensor[...] = 99
            return output

    class CreateNewBuffer:

        def _apply_unitary_(self, args):
            u = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=args.target_tensor.dtype) * 1j
            new_shape = args.target_tensor.shape[:-2] + (4, 1)
            ret = np.matmul(u, args.target_tensor.reshape(new_shape)).reshape(args.target_tensor.shape)
            args.target_tensor[...] = 99
            args.available_buffer[...] = 98
            return ret
    operations = [ModifyTargetTensor(), TransposeTargetTensor(), ReshapeTargetTensor(), ModifyAvailableBuffer(), TransposeAvailableBuffer(), ReshapeAvailableBuffer(), CreateNewBuffer()]

    def assert_is_swap_simple(val: cirq.SupportsConsistentApplyUnitary) -> None:
        qid_shape = (2, 2)
        op_indices = [0, 1]
        state = np.arange(3 * 3, dtype=np.complex64).reshape((1, 3, 3))
        expected = state.copy()
        buf = expected[..., 0, 1].copy()
        expected[..., 0, 1] = expected[..., 1, 0]
        expected[..., 1, 0] = buf
        expected[..., :2, :2] *= 1j
        args = cirq.ApplyUnitaryArgs(state, np.empty_like(state), [1, 2])
        sub_args = args._for_operation_with_qid_shape(op_indices, tuple((qid_shape[i] for i in op_indices)))
        sub_result = val._apply_unitary_(sub_args)
        result = _incorporate_result_into_target(args, sub_args, sub_result)
        np.testing.assert_allclose(result, expected, atol=1e-08)

    def assert_is_swap(val: cirq.SupportsConsistentApplyUnitary) -> None:
        qid_shape = (1, 2, 4, 2)
        op_indices = [1, 3]
        state = np.arange(2 * (1 * 3 * 4 * 5), dtype=np.complex64).reshape((1, 2, 1, 5, 3, 1, 4))
        expected = state.copy()
        buf = expected[..., 0, 1, :, :].copy()
        expected[..., 0, 1, :, :] = expected[..., 1, 0, :, :]
        expected[..., 1, 0, :, :] = buf
        expected[..., :2, :2, :, :] *= 1j
        args = cirq.ApplyUnitaryArgs(state, np.empty_like(state), [5, 4, 6, 3])
        sub_args = args._for_operation_with_qid_shape(op_indices, tuple((qid_shape[i] for i in op_indices)))
        sub_result = val._apply_unitary_(sub_args)
        result = _incorporate_result_into_target(args, sub_args, sub_result)
        np.testing.assert_allclose(result, expected, atol=1e-08, verbose=True)
    for op in operations:
        assert_is_swap_simple(op)
        assert_is_swap(op)