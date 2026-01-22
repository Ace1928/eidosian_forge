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
@pytest.mark.parametrize('circuit_cls', [cirq.Circuit, cirq.FrozenCircuit])
def test_measurement_key_mapping_preserves_moments(circuit_cls):
    a, b = cirq.LineQubit.range(2)
    c = circuit_cls(cirq.Moment(cirq.X(a)), cirq.Moment(), cirq.Moment(cirq.measure(a, key='m1')), cirq.Moment(cirq.measure(b, key='m2')))
    key_map = {'m1': 'p1'}
    remapped_circuit = cirq.with_measurement_key_mapping(c, key_map)
    assert list(remapped_circuit.moments) == [cirq.with_measurement_key_mapping(moment, key_map) for moment in c.moments]