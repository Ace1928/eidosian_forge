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
def test_octagonal_qubit_positions():
    """test OctagonalQubit 2D position and distance calculations"""
    qubit0 = OctagonalQubit(0)
    assert qubit0.octagon_position == 0
    assert qubit0.dimension == 2
    qubit5 = OctagonalQubit(5)
    assert qubit5.x == 0
    assert np.isclose(qubit5.y, 1 / sqrt(2))
    assert qubit5.z == 0
    qubit3 = OctagonalQubit(3)
    assert np.isclose(qubit3.x, 1 + 1 / sqrt(2))
    assert qubit3.y == 0
    assert qubit3.z == 0
    qubit2 = OctagonalQubit(2)
    assert np.isclose(qubit2.x, 1 + sqrt(2))
    assert np.isclose(qubit2.y, 1 / sqrt(2))
    assert qubit2.z == 0
    with patch('cirq_rigetti.OctagonalQubit.octagon_position', new_callable=PropertyMock) as mock_octagon_position:
        mock_octagon_position.return_value = 9
        invalid_qubit = OctagonalQubit(0)
        with pytest.raises(ValueError):
            _ = invalid_qubit.x
        with pytest.raises(ValueError):
            _ = invalid_qubit.y
    qubit0 = OctagonalQubit(0)
    assert np.isclose(qubit0.distance(OctagonalQubit(1)), 1)
    assert qubit0.distance(OctagonalQubit(7)) == 1
    with pytest.raises(TypeError):
        _ = qubit0.distance(AspenQubit(0, 0))