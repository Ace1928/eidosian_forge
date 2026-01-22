import pytest
import cirq
import numpy as np
from cirq.ops import QubitPermutationGate
def test_permutation_gate_diagram():
    q = cirq.LineQubit.range(6)
    cirq.testing.assert_has_diagram(cirq.Circuit(cirq.X(q[0]), cirq.X(q[5]), QubitPermutationGate([3, 2, 1, 0]).on(*q[1:5])), '\n0: ───X───────\n\n1: ───[0>3]───\n      │\n2: ───[1>2]───\n      │\n3: ───[2>1]───\n      │\n4: ───[3>0]───\n\n5: ───X───────\n')