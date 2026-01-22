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
@pytest.fixture
def qcs_aspen8_isa() -> InstructionSetArchitecture:
    with open(fixture_path / 'QCS-Aspen-8-ISA.json', 'r') as f:
        return InstructionSetArchitecture.from_dict(json.load(f))