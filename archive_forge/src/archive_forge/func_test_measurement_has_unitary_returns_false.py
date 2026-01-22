from typing import cast
import numpy as np
import pytest
import cirq
def test_measurement_has_unitary_returns_false():
    gate = cirq.MeasurementGate(1, 'a')
    assert not cirq.has_unitary(gate)