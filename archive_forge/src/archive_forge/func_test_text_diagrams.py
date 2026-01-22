import itertools
import pytest
import numpy as np
import sympy
import cirq
def test_text_diagrams():
    q0, q1 = (cirq.NamedQubit('q0'), cirq.NamedQubit('q1'))
    circuit = cirq.Circuit(cirq.PauliInteractionGate(cirq.X, False, cirq.X, False)(q0, q1), cirq.PauliInteractionGate(cirq.X, True, cirq.X, False)(q0, q1), cirq.PauliInteractionGate(cirq.X, False, cirq.X, True)(q0, q1), cirq.PauliInteractionGate(cirq.X, True, cirq.X, True)(q0, q1), cirq.PauliInteractionGate(cirq.X, False, cirq.Y, False)(q0, q1), cirq.PauliInteractionGate(cirq.Y, False, cirq.Z, False)(q0, q1), cirq.PauliInteractionGate(cirq.Z, False, cirq.Y, False)(q0, q1), cirq.PauliInteractionGate(cirq.Y, True, cirq.Z, True)(q0, q1), cirq.PauliInteractionGate(cirq.Z, True, cirq.Y, True)(q0, q1))
    assert circuit.to_text_diagram().strip() == '\nq0: ───X───(-X)───X──────(-X)───X───Y───@───(-Y)───(-@)───\n       │   │      │      │      │   │   │   │      │\nq1: ───X───X──────(-X)───(-X)───Y───@───Y───(-@)───(-Y)───\n    '.strip()