from typing import AbstractSet, Any, Dict, Optional, Tuple, TYPE_CHECKING, Union
import sympy
from cirq import value, protocols
from cirq.ops import raw_types
Initialize a wait gate with the given duration.

        Args:
            duration: A constant or parameterized wait duration. This can be
                an instance of `datetime.timedelta` or `cirq.Duration`.
            num_qubits: The number of qubits the gate operates on. If None and `qid_shape` is None,
                this defaults to one qubit.
            qid_shape: Can be specified instead of `num_qubits` for the case that the gate should
                act on qudits.

        Raises:
            ValueError: If the `qid_shape` provided is empty or `num_qubits` contradicts
                `qid_shape`.
        