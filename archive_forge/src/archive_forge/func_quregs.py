from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple
from numpy.typing import NDArray
import cirq
import numpy as np
from cirq._compat import cached_property
from cirq_ft.infra import gate_with_registers, t_complexity_protocol, merge_qubits, get_named_qubits
from cirq_ft.infra.decompose_protocol import _decompose_once_considering_known_decomposition
@cached_property
def quregs(self) -> Dict[str, NDArray[cirq.Qid]]:
    """A dictionary of named qubits appropriate for the signature for the gate."""
    return get_named_qubits(self.r)