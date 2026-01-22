import numpy as np
import pytest
import sympy
import cirq
def test_format_radians_with_precision():
    args = cirq.CircuitDiagramInfoArgs(known_qubits=None, known_qubit_count=None, use_unicode_characters=False, precision=3, label_map=None)
    assert args.format_radians(np.pi) == 'pi'
    assert args.format_radians(-np.pi) == '-pi'
    assert args.format_radians(np.pi / 2) == '0.5pi'
    assert args.format_radians(-3 * np.pi / 4) == '-0.75pi'
    assert args.format_radians(1.1) == '0.35pi'
    assert args.format_radians(1.234567) == '0.393pi'
    assert args.format_radians(sympy.Symbol('t')) == 't'
    assert args.format_radians(sympy.Symbol('t') * 2 + 1) == '2*t + 1'
    args.use_unicode_characters = True
    assert args.format_radians(np.pi) == 'π'
    assert args.format_radians(-np.pi) == '-π'
    assert args.format_radians(np.pi / 2) == '0.5π'
    assert args.format_radians(-3 * np.pi / 4) == '-0.75π'
    assert args.format_radians(1.1) == '0.35π'
    assert args.format_radians(1.234567) == '0.393π'
    assert args.format_radians(sympy.Symbol('t')) == 't'
    assert args.format_radians(sympy.Symbol('t') * 2 + 1) == '2*t + 1'