from typing import Iterator
import pytest
import sympy
import cirq
from cirq.study import sweeps
from cirq_google.study import DeviceParameter
from cirq_google.api import v2
@pytest.mark.parametrize('sweep', [cirq.UnitSweep, cirq.Linspace('a', 0, 10, 100), cirq.Linspace('a', 0, 10, 100, metadata=DeviceParameter(path=['path', 'to', 'parameter'], idx=2, units='ns')), cirq.Points('b', [1, 1.5, 2, 2.5, 3]), cirq.Points('b', [1, 1.5, 2, 2.5, 3], metadata=DeviceParameter(path=['path', 'to', 'parameter'], idx=2, units='GHz')), cirq.Points('b', [1, 1.5, 2, 2.5, 3], metadata=DeviceParameter(path=['path', 'to', 'parameter'], idx=None)), cirq.Linspace('a', 0, 1, 5) * cirq.Linspace('b', 0, 1, 5), cirq.Points('a', [1, 2, 3]) + cirq.Linspace('b', 0, 1, 3), cirq.Linspace('a', 0, 1, 3) * (cirq.Linspace('b', 0, 1, 4) + cirq.Linspace('c', 0, 10, 4)) * (cirq.Linspace('d', 0, 1, 5) + cirq.Linspace('e', 0, 10, 5)) * (cirq.Linspace('f', 0, 1, 6) + cirq.Points('g', [1, 2]) * cirq.Points('h', [-1, 0, 1]))])
def test_sweep_to_proto_roundtrip(sweep):
    msg = v2.sweep_to_proto(sweep)
    deserialized = v2.sweep_from_proto(msg)
    assert deserialized == sweep
    assert getattr(deserialized, 'metadata', None) == getattr(sweep, 'metadata', None)