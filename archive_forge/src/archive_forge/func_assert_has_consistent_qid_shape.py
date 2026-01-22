from typing import Any, Dict, Iterable, List, Optional, Sequence, Union
from collections import defaultdict
import itertools
import random
import numpy as np
import sympy
from cirq import circuits, ops, linalg, protocols, qis
from cirq.testing import lin_alg_utils
def assert_has_consistent_qid_shape(val: Any) -> None:
    """Tests whether a value's `_qid_shape_` and `_num_qubits_` are correct and
    consistent.

    Verifies that the entries in the shape are all positive integers and the
    length of shape equals `_num_qubits_` (and also equals `len(qubits)` if
    `val` has `qubits`.

    Args:
        val: The value under test. Should have `_qid_shape_` and/or
            `num_qubits_` methods. Can optionally have a `qubits` property.
    """
    __tracebackhide__ = True
    default = (-1,)
    qid_shape = protocols.qid_shape(val, default)
    num_qubits = protocols.num_qubits(val, default)
    if qid_shape is default or num_qubits is default:
        return
    assert all((d >= 1 for d in qid_shape)), f'Not all entries in qid_shape are positive: {qid_shape}'
    assert len(qid_shape) == num_qubits, f'Length of qid_shape and num_qubits disagree: {qid_shape}, {num_qubits}'
    if isinstance(val, ops.Operation):
        assert num_qubits == len(val.qubits), f'Length of num_qubits and val.qubits disagrees: {num_qubits}, {len(val.qubits)}'