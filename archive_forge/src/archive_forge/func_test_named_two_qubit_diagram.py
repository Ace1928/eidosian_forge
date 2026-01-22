import re
import numpy as np
import pytest
import sympy
import cirq
def test_named_two_qubit_diagram():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    c = cirq.NamedQubit('c')
    c = cirq.Circuit(cirq.MatrixGate(cirq.unitary(cirq.CZ), name='Foo').on(a, b), cirq.MatrixGate(cirq.unitary(cirq.CZ), name='Bar').on(c, a))
    expected_horizontal = '\na: ───Foo[1]───Bar[2]───\n      │        │\nb: ───Foo[2]───┼────────\n               │\nc: ────────────Bar[1]───\n    '.strip()
    assert expected_horizontal == c.to_text_diagram().strip()
    expected_vertical = '\na      b      c\n│      │      │\nFoo[1]─Foo[2] │\n│      │      │\nBar[2]─┼──────Bar[1]\n│      │      │\n    '.strip()
    assert expected_vertical == c.to_text_diagram(transpose=True).strip()