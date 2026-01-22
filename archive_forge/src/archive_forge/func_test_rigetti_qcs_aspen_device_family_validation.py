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
def test_rigetti_qcs_aspen_device_family_validation(qcs_aspen8_isa: InstructionSetArchitecture):
    """test RigettiQCSAspenDevice validates architecture family on initialization"""
    non_aspen_isa = InstructionSetArchitecture.from_dict(qcs_aspen8_isa.to_dict())
    non_aspen_isa.architecture.family = 'not-aspen'
    with pytest.raises(UnsupportedRigettiQCSQuantumProcessor):
        RigettiQCSAspenDevice(isa=non_aspen_isa)