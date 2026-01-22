from typing import Iterator
import pytest
import sympy
import cirq
from cirq.study import sweeps
from cirq_google.study import DeviceParameter
from cirq_google.api import v2
def test_sweep_from_proto_single_sweep_type_not_set():
    proto = v2.run_context_pb2.Sweep()
    proto.single_sweep.parameter_key = 'foo'
    with pytest.raises(ValueError, match='single sweep type not set'):
        v2.sweep_from_proto(proto)