import os
from unittest.mock import patch, PropertyMock
from math import sqrt
import pathlib
import json
import pytest
import cirq
from cirq_rigetti import (
from qcs_api_client.models import InstructionSetArchitecture, Node
import numpy as np
@pytest.mark.parametrize('qubit', [cirq.GridQubit(2, 2), cirq.LineQubit(33), cirq.NamedQubit('s'), cirq.NamedQubit('40'), cirq.NamedQubit('9'), AspenQubit(4, 0)])
def test_rigetti_qcs_aspen_device_invalid_qubit(qubit: cirq.Qid, qcs_aspen8_isa: InstructionSetArchitecture):
    """test RigettiQCSAspenDevice throws error on invalid qubits"""
    device = RigettiQCSAspenDevice(isa=qcs_aspen8_isa)
    with pytest.raises(UnsupportedQubit):
        device.validate_qubit(qubit)
    with pytest.raises(UnsupportedQubit):
        device.validate_operation(cirq.I(qubit))