import pytest
import sympy
import numpy as np
import cirq
import cirq_pasqal
from cirq_pasqal import PasqalDevice, PasqalVirtualDevice
from cirq_pasqal import TwoDQubit, ThreeDQubit
def test_value_equal():
    dev = PasqalDevice(qubits=[cirq.NamedQubit('q1')])
    assert PasqalDevice(qubits=[cirq.NamedQubit('q1')]) == dev
    dev = PasqalVirtualDevice(control_radius=1.0, qubits=[TwoDQubit(0, 0)])
    assert PasqalVirtualDevice(control_radius=1.0, qubits=[TwoDQubit(0, 0)]) == dev