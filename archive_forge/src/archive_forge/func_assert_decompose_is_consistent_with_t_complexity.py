from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple
from numpy.typing import NDArray
import cirq
import numpy as np
from cirq._compat import cached_property
from cirq_ft.infra import gate_with_registers, t_complexity_protocol, merge_qubits, get_named_qubits
from cirq_ft.infra.decompose_protocol import _decompose_once_considering_known_decomposition
def assert_decompose_is_consistent_with_t_complexity(val: Any):
    t_complexity_method = getattr(val, '_t_complexity_', None)
    expected = NotImplemented if t_complexity_method is None else t_complexity_method()
    if expected is NotImplemented or expected is None:
        return
    decomposition = _decompose_once_considering_known_decomposition(val)
    if decomposition is None:
        return
    from_decomposition = t_complexity_protocol.t_complexity(decomposition, fail_quietly=False)
    assert expected == from_decomposition, f'{expected} != {from_decomposition}'