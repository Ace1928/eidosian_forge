import itertools
from typing import Optional
from unittest import mock
import pytest
import cirq
def use_qubits_twice(*qubits):
    a = list(qubits)
    b = list(qubits)
    yield cirq.X.on_each(*a)
    yield cirq.Y.on_each(*b)