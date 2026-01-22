from typing import Tuple, Optional, List, Union, Generic, TypeVar, Dict
from unittest.mock import create_autospec, Mock
import pytest
from pyquil import Program
from pyquil.quantum_processor import AbstractQuantumProcessor, NxQuantumProcessor
from pyquil.api import QAM, QuantumComputer, QuantumExecutable, QAMExecutionResult, EncryptedProgram
from pyquil.api._abstract_compiler import AbstractCompiler
from qcs_api_client.client._configuration.settings import QCSClientConfigurationSettings
from qcs_api_client.client._configuration import (
import networkx as nx
import cirq
import sympy
import numpy as np
@pytest.fixture
def bell_circuit_with_qids() -> Tuple[cirq.Circuit, List[cirq.LineQubit]]:
    bell_circuit = cirq.Circuit()
    qubits = cirq.LineQubit.range(2)
    bell_circuit.append(cirq.H(qubits[0]))
    bell_circuit.append(cirq.CNOT(qubits[0], qubits[1]))
    bell_circuit.append(cirq.measure(qubits[0], qubits[1], key='m'))
    return (bell_circuit, qubits)