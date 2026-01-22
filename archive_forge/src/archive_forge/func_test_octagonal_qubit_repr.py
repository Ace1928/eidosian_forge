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
def test_octagonal_qubit_repr():
    """test OctagonalQubit.__repr__"""
    qubit5 = OctagonalQubit(5)
    assert 'cirq_rigetti.OctagonalQubit(octagon_position=5)' == repr(qubit5)