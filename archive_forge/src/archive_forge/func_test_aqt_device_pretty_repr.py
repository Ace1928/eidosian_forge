from datetime import timedelta
from typing import List
import pytest
import cirq
from cirq_aqt import aqt_device, aqt_device_metadata
def test_aqt_device_pretty_repr(device):
    cirq.testing.assert_repr_pretty(device, 'q(0)───q(1)───q(2)')
    cirq.testing.assert_repr_pretty(device, 'AQTDevice(...)', cycle=True)