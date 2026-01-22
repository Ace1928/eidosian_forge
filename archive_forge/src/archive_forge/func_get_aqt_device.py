import json
from typing import Any, cast, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union
import networkx as nx
import numpy as np
import cirq
from cirq_aqt import aqt_device_metadata
def get_aqt_device(num_qubits: int) -> Tuple[AQTDevice, List[cirq.LineQubit]]:
    """Returns an AQT ion device

    Args:
        num_qubits: number of qubits

    Returns:
         A tuple of AQTDevice and qubit_list
    """
    qubit_list = cirq.LineQubit.range(num_qubits)
    us = 1000 * cirq.Duration(nanos=1)
    ion_device = AQTDevice(measurement_duration=100 * us, twoq_gates_duration=200 * us, oneq_gates_duration=10 * us, qubits=qubit_list)
    return (ion_device, qubit_list)