import numpy as np
import pytest
import sympy
from scipy import linalg
import cirq
def test_iswap_decompose_diagram():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    decomposed = cirq.Circuit(cirq.decompose_once(cirq.ISWAP(a, b) ** 0.5))
    cirq.testing.assert_has_diagram(decomposed, '\na: ───@───H───X───T───X───T^-1───H───@───\n      │       │       │              │\nb: ───X───────@───────@──────────────X───\n')