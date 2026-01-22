from typing import Iterator
import pytest
import sympy
import cirq
from cirq.study import sweeps
from cirq_google.study import DeviceParameter
from cirq_google.api import v2
def test_sweep_to_proto_linspace():
    proto = v2.sweep_to_proto(cirq.Linspace('foo', 0, 1, 20, metadata=DeviceParameter(path=['path', 'to', 'parameter'], idx=2)))
    assert isinstance(proto, v2.run_context_pb2.Sweep)
    assert proto.HasField('single_sweep')
    assert proto.single_sweep.parameter_key == 'foo'
    assert proto.single_sweep.WhichOneof('sweep') == 'linspace'
    assert proto.single_sweep.linspace.first_point == 0
    assert proto.single_sweep.linspace.last_point == 1
    assert proto.single_sweep.linspace.num_points == 20
    assert proto.single_sweep.parameter.path == ['path', 'to', 'parameter']
    assert proto.single_sweep.parameter.idx == 2
    assert v2.sweep_from_proto(proto).metadata == DeviceParameter(path=['path', 'to', 'parameter'], idx=2)