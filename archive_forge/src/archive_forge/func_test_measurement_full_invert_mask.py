from typing import cast
import numpy as np
import pytest
import cirq
def test_measurement_full_invert_mask():
    assert cirq.MeasurementGate(1, 'a').full_invert_mask() == (False,)
    assert cirq.MeasurementGate(2, 'a', invert_mask=(False, True)).full_invert_mask() == (False, True)
    assert cirq.MeasurementGate(2, 'a', invert_mask=(True,)).full_invert_mask() == (True, False)