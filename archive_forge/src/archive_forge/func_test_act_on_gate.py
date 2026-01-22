import itertools
import math
import numpy as np
import pytest
import sympy
import cirq
import cirq.contrib.quimb as ccq
import cirq.testing
from cirq import value
def test_act_on_gate():
    args = ccq.mps_simulator.MPSState(qubits=cirq.LineQubit.range(3), prng=np.random.RandomState(0))
    cirq.act_on(cirq.X, args, [cirq.LineQubit(1)])
    np.testing.assert_allclose(args.state_vector().reshape((2, 2, 2)), cirq.one_hot(index=(0, 1, 0), shape=(2, 2, 2), dtype=np.complex64))