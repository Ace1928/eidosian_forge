import re
import numpy as np
import pytest
import sympy
import cirq
def test_single_qubit_diagram():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    m = np.array([[1, 1j], [1j, 1]]) * np.sqrt(0.5)
    c = cirq.Circuit(cirq.MatrixGate(m).on(a), cirq.CZ(a, b))
    assert re.match('\n      ┌[          ]+┐\na: ───│[0-9\\.+\\-j ]+│───@───\n      │[0-9\\.+\\-j ]+│   │\n      └[          ]+┘   │\n       [          ]+    │\nb: ────[──────────]+────@───\n    '.strip(), c.to_text_diagram().strip())
    assert re.match('\na[          ]+  b\n│[          ]+  │\n┌[          ]+┐ │\n│[0-9\\.+\\-j ]+│ │\n│[0-9\\.+\\-j ]+│ │\n└[          ]+┘ │\n│[          ]+  │\n@[──────────]+──@\n│[          ]+  │\n    '.strip(), c.to_text_diagram(transpose=True).strip())