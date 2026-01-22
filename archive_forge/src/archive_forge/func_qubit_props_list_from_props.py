from __future__ import annotations
import logging
import warnings
from typing import List, Iterable, Any, Dict, Optional
from qiskit.providers.backend import BackendV1, BackendV2
from qiskit.providers.backend import QubitProperties
from qiskit.providers.models.backendconfiguration import BackendConfiguration
from qiskit.providers.models.backendproperties import BackendProperties
from qiskit.circuit.controlflow import CONTROL_FLOW_OP_NAMES
from qiskit.providers.models.pulsedefaults import PulseDefaults
from qiskit.providers.options import Options
from qiskit.providers.exceptions import BackendPropertyError
def qubit_props_list_from_props(properties: BackendProperties) -> List[QubitProperties]:
    """Uses BackendProperties to construct
    and return a list of QubitProperties.
    """
    qubit_props: List[QubitProperties] = []
    for qubit, _ in enumerate(properties.qubits):
        try:
            t_1 = properties.t1(qubit)
        except BackendPropertyError:
            t_1 = None
        try:
            t_2 = properties.t2(qubit)
        except BackendPropertyError:
            t_2 = None
        try:
            frequency = properties.frequency(qubit)
        except BackendPropertyError:
            frequency = None
        qubit_props.append(QubitProperties(t1=t_1, t2=t_2, frequency=frequency))
    return qubit_props