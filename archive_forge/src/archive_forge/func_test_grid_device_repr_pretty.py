from dataclasses import dataclass
from typing import Dict, List, Tuple
import unittest.mock as mock
import pytest
import cirq
import cirq_google
from cirq_google.api import v2
from cirq_google.devices import grid_device
@pytest.mark.parametrize('cycle,func', [(False, str), (True, repr)])
def test_grid_device_repr_pretty(cycle, func):
    spec = _create_device_spec_with_all_couplings()
    device = cirq_google.GridDevice.from_proto(spec)
    printer = mock.Mock()
    device._repr_pretty_(printer, cycle)
    printer.text.assert_called_once_with(func(device))