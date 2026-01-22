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
def test_rigetti_qcs_aspen_device_non_existent_qubit(qcs_aspen8_isa: InstructionSetArchitecture):
    """test RigettiQCSAspenDevice throws error when qubit does not exist on device"""
    device_with_limited_nodes = RigettiQCSAspenDevice(isa=InstructionSetArchitecture.from_dict(qcs_aspen8_isa.to_dict()))
    device_with_limited_nodes.isa.architecture.nodes = [Node(node_id=10)]
    with pytest.raises(UnsupportedQubit):
        device_with_limited_nodes.validate_qubit(cirq.GridQubit(0, 0))