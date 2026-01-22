from itertools import combinations, product
from random import randint
from string import ascii_lowercase as alphabet
from typing import Optional, Sequence, Tuple
import numpy
import pytest
import cirq
import cirq.testing as ct
import cirq.contrib.acquaintance as cca
def test_acquaintance_gate_repr():
    assert repr(cca.AcquaintanceOpportunityGate(2)) == 'cirq.contrib.acquaintance.AcquaintanceOpportunityGate(num_qubits=2)'