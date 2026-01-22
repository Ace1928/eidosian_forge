import numpy as np
import pytest
import sympy
import cirq
def test_format_radians_without_precision():
    args = cirq.CircuitDiagramInfoArgs(known_qubits=None, known_qubit_count=None, use_unicode_characters=False, precision=None, label_map=None)
    assert args.format_radians(np.pi) == 'pi'
    assert args.format_radians(-np.pi) == '-pi'
    assert args.format_radians(1.1) == '1.1'
    assert args.format_radians(1.234567) == '1.234567'
    assert args.format_radians(1 / 7) == repr(1 / 7)
    assert args.format_radians(sympy.Symbol('t')) == 't'
    assert args.format_radians(sympy.Symbol('t') * 2 + 1) == '2*t + 1'
    args.use_unicode_characters = True
    assert args.format_radians(np.pi) == 'π'
    assert args.format_radians(-np.pi) == '-π'
    assert args.format_radians(1.1) == '1.1'
    assert args.format_radians(1.234567) == '1.234567'
    assert args.format_radians(1 / 7) == repr(1 / 7)
    assert args.format_radians(sympy.Symbol('t')) == 't'
    assert args.format_radians(sympy.Symbol('t') * 2 + 1) == '2*t + 1'