import pytest
import sympy
import numpy as np
import cirq
import cirq_pasqal
from cirq_pasqal import PasqalDevice, PasqalVirtualDevice
from cirq_pasqal import TwoDQubit, ThreeDQubit
def test_validate_operation_errors():
    d = generic_device(3)
    with pytest.raises(ValueError, match='Unsupported operation'):
        d.validate_operation(cirq.NamedQubit('q0'))
    with pytest.raises(ValueError, match='is not a supported gate'):
        d.validate_operation((cirq.ops.H ** 0.2).on(cirq.NamedQubit('q0')))
    with pytest.raises(ValueError, match='is not a valid qubit for gate cirq.X'):
        d.validate_operation(cirq.X.on(cirq.LineQubit(0)))
    with pytest.raises(ValueError, match='is not part of the device.'):
        d.validate_operation(cirq.X.on(cirq.NamedQubit('q6')))
    d = square_virtual_device(control_r=1.0, num_qubits=3)
    with pytest.raises(ValueError, match='are too far away'):
        d.validate_operation(cirq.CZ.on(TwoDQubit(0, 0), TwoDQubit(2, 2)))