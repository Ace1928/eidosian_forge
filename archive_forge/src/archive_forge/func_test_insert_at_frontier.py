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
def test_insert_at_frontier():

    class Replacer(cirq.PointOptimizer):

        def __init__(self, replacer=lambda x: x):
            super().__init__()
            self.replacer = replacer

        def optimization_at(self, circuit: 'cirq.Circuit', index: int, op: 'cirq.Operation') -> Optional['cirq.PointOptimizationSummary']:
            new_ops = self.replacer(op)
            return cirq.PointOptimizationSummary(clear_span=1, clear_qubits=op.qubits, new_operations=new_ops)
    replacer = lambda op: (cirq.Z(op.qubits[0]),) * 2 + (op, cirq.Y(op.qubits[0]))
    prepend_two_Xs_append_one_Y = Replacer(replacer)
    qubits = [cirq.NamedQubit(s) for s in 'abcdef']
    a, b, c = qubits[:3]
    circuit = cirq.Circuit([cirq.Moment([cirq.CZ(a, b)]), cirq.Moment([cirq.CZ(b, c)]), cirq.Moment([cirq.CZ(a, b)])])
    prepend_two_Xs_append_one_Y.optimize_circuit(circuit)
    cirq.testing.assert_has_diagram(circuit, '\na: ───Z───Z───@───Y───────────────Z───Z───@───Y───\n              │                           │\nb: ───────────@───Z───Z───@───Y───────────@───────\n                          │\nc: ───────────────────────@───────────────────────\n')
    prepender = lambda op: (cirq.X(op.qubits[0]),) * 3 + (op,)
    prepend_3_Xs = Replacer(prepender)
    circuit = cirq.Circuit([cirq.Moment([cirq.CNOT(a, b)]), cirq.Moment([cirq.CNOT(b, c)]), cirq.Moment([cirq.CNOT(c, b)])])
    prepend_3_Xs.optimize_circuit(circuit)
    cirq.testing.assert_has_diagram(circuit, '\na: ───X───X───X───@───────────────────────────────────\n                  │\nb: ───────────────X───X───X───X───@───────────────X───\n                                  │               │\nc: ───────────────────────────────X───X───X───X───@───\n')
    duplicate = Replacer(lambda op: (op,) * 2)
    circuit = cirq.Circuit([cirq.Moment([cirq.CZ(qubits[j], qubits[j + 1]) for j in range(i % 2, 5, 2)]) for i in range(4)])
    duplicate.optimize_circuit(circuit)
    cirq.testing.assert_has_diagram(circuit, '\na: ───@───@───────────@───@───────────\n      │   │           │   │\nb: ───@───@───@───@───@───@───@───@───\n              │   │           │   │\nc: ───@───@───@───@───@───@───@───@───\n      │   │           │   │\nd: ───@───@───@───@───@───@───@───@───\n              │   │           │   │\ne: ───@───@───@───@───@───@───@───@───\n      │   │           │   │\nf: ───@───@───────────@───@───────────\n')
    circuit = cirq.Circuit([cirq.Moment([cirq.CZ(*qubits[2:4]), cirq.CNOT(*qubits[:2])]), cirq.Moment([cirq.CNOT(*qubits[1::-1])])])
    duplicate.optimize_circuit(circuit)
    cirq.testing.assert_has_diagram(circuit, '\na: ───@───@───X───X───\n      │   │   │   │\nb: ───X───X───@───@───\n\nc: ───@───────@───────\n      │       │\nd: ───@───────@───────\n')