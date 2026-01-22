import pytest
import sympy
import numpy as np
import cirq
import cirq_pasqal
from cirq_pasqal import PasqalDevice, PasqalVirtualDevice
from cirq_pasqal import TwoDQubit, ThreeDQubit
def square_virtual_device(control_r, num_qubits) -> PasqalVirtualDevice:
    return PasqalVirtualDevice(control_radius=control_r, qubits=TwoDQubit.square(num_qubits))