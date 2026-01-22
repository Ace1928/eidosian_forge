import pytest
import numpy as np
import sympy
import cirq
def test_with_num_copies():
    g = cirq.testing.SingleQubitGate()
    pg = cirq.ParallelGate(g, 3)
    assert pg.with_num_copies(5) == cirq.ParallelGate(g, 5)