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
def test_append_control_key_subcircuit():
    q0, q1 = cirq.LineQubit.range(2)
    c = cirq.Circuit()
    c.append(cirq.measure(q0, key='a'))
    c.append(cirq.CircuitOperation(cirq.FrozenCircuit(cirq.ClassicallyControlledOperation(cirq.X(q1), 'a'))))
    assert len(c) == 2
    c = cirq.Circuit()
    c.append(cirq.measure(q0, key='a'))
    c.append(cirq.CircuitOperation(cirq.FrozenCircuit(cirq.ClassicallyControlledOperation(cirq.X(q1), 'b'))))
    assert len(c) == 1
    c = cirq.Circuit()
    c.append(cirq.measure(q0, key='a'))
    c.append(cirq.CircuitOperation(cirq.FrozenCircuit(cirq.ClassicallyControlledOperation(cirq.X(q1), 'b'))).with_measurement_key_mapping({'b': 'a'}))
    assert len(c) == 2
    c = cirq.Circuit()
    c.append(cirq.CircuitOperation(cirq.FrozenCircuit(cirq.measure(q0, key='a'))))
    c.append(cirq.CircuitOperation(cirq.FrozenCircuit(cirq.ClassicallyControlledOperation(cirq.X(q1), 'b'))).with_measurement_key_mapping({'b': 'a'}))
    assert len(c) == 2
    c = cirq.Circuit()
    c.append(cirq.CircuitOperation(cirq.FrozenCircuit(cirq.measure(q0, key='a'))).with_measurement_key_mapping({'a': 'c'}))
    c.append(cirq.CircuitOperation(cirq.FrozenCircuit(cirq.ClassicallyControlledOperation(cirq.X(q1), 'b'))).with_measurement_key_mapping({'b': 'c'}))
    assert len(c) == 2
    c = cirq.Circuit()
    c.append(cirq.CircuitOperation(cirq.FrozenCircuit(cirq.measure(q0, key='a'))).with_measurement_key_mapping({'a': 'b'}))
    c.append(cirq.CircuitOperation(cirq.FrozenCircuit(cirq.ClassicallyControlledOperation(cirq.X(q1), 'b'))).with_measurement_key_mapping({'b': 'a'}))
    assert len(c) == 1