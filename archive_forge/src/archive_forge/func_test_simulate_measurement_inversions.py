import itertools
import random
from typing import Type
from unittest import mock
import numpy as np
import pytest
import sympy
import cirq
def test_simulate_measurement_inversions():
    q = cirq.NamedQubit('q')
    c = cirq.Circuit(cirq.measure(q, key='q', invert_mask=(True,)))
    assert cirq.Simulator().simulate(c).measurements == {'q': np.array([True])}
    c = cirq.Circuit(cirq.measure(q, key='q', invert_mask=(False,)))
    assert cirq.Simulator().simulate(c).measurements == {'q': np.array([False])}