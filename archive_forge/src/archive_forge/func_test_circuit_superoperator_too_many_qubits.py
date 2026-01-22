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
def test_circuit_superoperator_too_many_qubits():
    circuit = cirq.Circuit(cirq.IdentityGate(num_qubits=11).on(*cirq.LineQubit.range(11)))
    assert not circuit._has_superoperator_()
    with pytest.raises(ValueError, match='too many'):
        _ = circuit._superoperator_()