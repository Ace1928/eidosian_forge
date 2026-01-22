from typing import Any, TypeVar, Optional
import numpy as np
from typing_extensions import Protocol
from cirq import qis
from cirq._doc import doc_private
from cirq.protocols import qid_shape_protocol
from cirq.protocols.apply_unitary_protocol import ApplyUnitaryArgs
from cirq.protocols.decompose_protocol import _try_decompose_into_operations_and_qubits
Determines whether the receiver has a unitary effect.

        This method is used preferentially by the global `cirq.has_unitary`
        method, because this method is much cheaper than the fallback strategies
        such as checking `value._unitary_` (which causes a large matrix to be
        computed).

        Returns:
            Whether or not the receiving object (`self`) has a unitary effect.
        