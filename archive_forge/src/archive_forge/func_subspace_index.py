import warnings
from typing import Any, cast, Iterable, Optional, Sequence, Tuple, TYPE_CHECKING, TypeVar, Union
import numpy as np
from typing_extensions import Protocol
from cirq import linalg, qis
from cirq._doc import doc_private
from cirq.protocols import qid_shape_protocol
from cirq.protocols.decompose_protocol import _try_decompose_into_operations_and_qubits
from cirq.type_workarounds import NotImplementedType
def subspace_index(self, little_endian_bits_int: int=0, *, big_endian_bits_int: int=0) -> Tuple[Union[slice, int, 'ellipsis'], ...]:
    """An index for the subspace where the target axes equal a value.

        Args:
            little_endian_bits_int: The desired value of the qubits at the
                targeted `axes`, packed into an integer. The least significant
                bit of the integer is the desired bit for the first axis, and
                so forth in increasing order. Can't be specified at the same
                time as `big_endian_bits_int`.
            big_endian_bits_int: The desired value of the qubits at the
                targeted `axes`, packed into an integer. The most significant
                bit of the integer is the desired bit for the first axis, and
                so forth in decreasing order. Can't be specified at the same
                time as `little_endian_bits_int`.

        Returns:
            A value that can be used to index into `target_tensor` and
            `available_buffer`, and manipulate only the part of Hilbert space
            corresponding to a given bit assignment.

        Example:
            If `target_tensor` is a 4 qubit tensor and `axes` is `[1, 3]` and
            then this method will return the following when given
            `little_endian_bits=0b01`:

                `(slice(None), 0, slice(None), 1, Ellipsis)`

            Therefore the following two lines would be equivalent:

                args.target_tensor[args.subspace_index(0b01)] += 1

                args.target_tensor[:, 0, :, 1] += 1
        """
    return linalg.slice_for_qubits_equal_to(self.axes, little_endian_qureg_value=little_endian_bits_int, big_endian_qureg_value=big_endian_bits_int, qid_shape=self.target_tensor.shape)