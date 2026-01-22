import itertools
import os
import time
from collections import defaultdict
from random import randint, random, sample, randrange
from typing import Iterator, Optional, Tuple, TYPE_CHECKING
import numpy as np
import pytest
import sympy
import cirq
from cirq import circuits
from cirq import ops
from cirq.testing.devices import ValidatingTestDevice
@pytest.mark.parametrize('circuit, mutate', [(cirq.Circuit(cirq.X(cirq.q(0)), cirq.M(cirq.q(0))), lambda c: c.__setitem__(0, cirq.Moment(cirq.Y(cirq.q(0))))), (cirq.Circuit(cirq.X(cirq.q(0)), cirq.M(cirq.q(0))), lambda c: c.__delitem__(0)), (cirq.Circuit(cirq.X(cirq.q(0)), cirq.M(cirq.q(0))), lambda c: c.__imul__(2)), (cirq.Circuit(cirq.X(cirq.q(0)), cirq.M(cirq.q(0))), lambda c: c.insert(1, cirq.Y(cirq.q(0)))), (cirq.Circuit(cirq.X(cirq.q(0)), cirq.M(cirq.q(0))), lambda c: c.insert_into_range([cirq.Y(cirq.q(1)), cirq.M(cirq.q(1))], 0, 2)), (cirq.Circuit(cirq.X(cirq.q(0)), cirq.M(cirq.q(0))), lambda c: c.insert_at_frontier([cirq.Y(cirq.q(0)), cirq.Y(cirq.q(1))], 1)), (cirq.Circuit(cirq.X(cirq.q(0)), cirq.M(cirq.q(0))), lambda c: c.batch_replace([(0, cirq.X(cirq.q(0)), cirq.Y(cirq.q(0)))])), (cirq.Circuit(cirq.X(cirq.q(0)), cirq.M(cirq.q(0), cirq.q(1))), lambda c: c.batch_insert_into([(0, cirq.X(cirq.q(1)))])), (cirq.Circuit(cirq.X(cirq.q(0)), cirq.M(cirq.q(0))), lambda c: c.batch_insert([(1, cirq.Y(cirq.q(0)))])), (cirq.Circuit(cirq.X(cirq.q(0)), cirq.M(cirq.q(0))), lambda c: c.clear_operations_touching([cirq.q(0)], [0]))])
def test_mutation_clears_cached_attributes(circuit, mutate):
    cached_attributes = ['_all_qubits', '_frozen', '_is_measurement', '_is_parameterized', '_parameter_names']
    for attr in cached_attributes:
        assert getattr(circuit, attr) is None, f'attr={attr!r} is not None'
    qubits = circuit.all_qubits()
    frozen = circuit.freeze()
    is_measurement = cirq.is_measurement(circuit)
    is_parameterized = cirq.is_parameterized(circuit)
    parameter_names = cirq.parameter_names(circuit)
    for attr in cached_attributes:
        assert getattr(circuit, attr) is not None, f'attr={attr!r} is None'
    assert circuit.all_qubits() is qubits
    assert circuit.freeze() is frozen
    assert cirq.is_measurement(circuit) is is_measurement
    assert cirq.is_parameterized(circuit) is is_parameterized
    assert cirq.parameter_names(circuit) is parameter_names
    mutate(circuit)
    for attr in cached_attributes:
        assert getattr(circuit, attr) is None, f'attr={attr!r} is not None'