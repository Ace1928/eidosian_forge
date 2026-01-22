from typing import List
import datetime
import pytest
import numpy as np
import sympy
import cirq
import cirq_google
from cirq_google.api import v2
from cirq_google.cloud import quantum
from cirq_google.engine.simulated_local_processor import SimulatedLocalProcessor, VALID_LANGUAGES
def test_device_specification():
    proc = SimulatedLocalProcessor(processor_id='test_proc')
    assert proc.get_device_specification() is None
    device_spec = v2.device_pb2.DeviceSpecification()
    device_spec.valid_qubits.append('q0_0')
    device_spec.valid_qubits.append('q0_1')
    proc = SimulatedLocalProcessor(processor_id='test_proc', device_specification=device_spec)
    assert proc.get_device_specification() == device_spec