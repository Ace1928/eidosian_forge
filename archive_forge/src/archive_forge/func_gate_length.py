import copy
import datetime
from typing import Any, Iterable, Tuple, Union, Dict
import dateutil.parser
from qiskit.providers.exceptions import BackendPropertyError
from qiskit.utils.units import apply_prefix
def gate_length(self, gate: str, qubits: Union[int, Iterable[int]]) -> float:
    """
        Return the duration of the gate in units of seconds.

        Args:
            gate: The gate for which to get the duration.
            qubits: The specific qubits for the gate.

        Returns:
            Gate length of the given gate and qubit(s).
        """
    return self.gate_property(gate, qubits, 'gate_length')[0]