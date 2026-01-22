import warnings
from typing import Any, cast, Iterable, Optional, Sequence, Tuple, TYPE_CHECKING, TypeVar, Union
import numpy as np
from typing_extensions import Protocol
from cirq import linalg, qis
from cirq._doc import doc_private
from cirq.protocols import qid_shape_protocol
from cirq.protocols.decompose_protocol import _try_decompose_into_operations_and_qubits
from cirq.type_workarounds import NotImplementedType
def with_axes_transposed_to_start(self) -> 'ApplyUnitaryArgs':
    """Returns a transposed view of the same arguments.

        Returns:
            A view over the same target tensor and available workspace, but
            with the numpy arrays transposed such that the axes field is
            guaranteed to equal `range(len(result.axes))`. This allows one to
            say e.g. `result.target_tensor[0, 1, 0, ...]` instead of
            `result.target_tensor[result.subspace_index(0b010)]`.
        """
    axis_set = set(self.axes)
    other_axes = [axis for axis in range(len(self.target_tensor.shape)) if axis not in axis_set]
    perm = (*self.axes, *other_axes)
    target_tensor = self.target_tensor.transpose(*perm)
    available_buffer = self.available_buffer.transpose(*perm)
    return ApplyUnitaryArgs(target_tensor, available_buffer, range(len(self.axes)))