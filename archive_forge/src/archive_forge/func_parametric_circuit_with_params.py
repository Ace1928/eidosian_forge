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
def parametric_circuit_with_params() -> Tuple[cirq.Circuit, cirq.Linspace]:
    q = cirq.GridQubit(1, 1)
    circuit = cirq.Circuit(cirq.X(q) ** sympy.Symbol('t'), cirq.measure(q, key='m'))
    param_sweep = cirq.Linspace('t', start=0, stop=2, length=5)
    return (circuit, param_sweep)