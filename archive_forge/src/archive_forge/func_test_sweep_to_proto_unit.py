from typing import Iterator
import pytest
import sympy
import cirq
from cirq.study import sweeps
from cirq_google.study import DeviceParameter
from cirq_google.api import v2
def test_sweep_to_proto_unit():
    proto = v2.sweep_to_proto(cirq.UnitSweep)
    assert isinstance(proto, v2.run_context_pb2.Sweep)
    assert not proto.HasField('single_sweep')
    assert not proto.HasField('sweep_function')