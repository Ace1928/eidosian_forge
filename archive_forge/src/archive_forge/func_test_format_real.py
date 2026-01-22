import numpy as np
import pytest
import sympy
import cirq
def test_format_real():
    args = cirq.CircuitDiagramInfoArgs.UNINFORMED_DEFAULT.copy()
    assert args.format_real(1) == '1'
    assert args.format_real(1.1) == '1.1'
    assert args.format_real(1.234567) == '1.23'
    assert args.format_real(1 / 7) == '0.143'
    assert args.format_real(sympy.Symbol('t')) == 't'
    assert args.format_real(sympy.Symbol('t') * 2 + 1) == '2*t + 1'
    args.precision = None
    assert args.format_real(1) == '1'
    assert args.format_real(1.1) == '1.1'
    assert args.format_real(1.234567) == '1.234567'
    assert args.format_real(1 / 7) == repr(1 / 7)
    assert args.format_real(sympy.Symbol('t')) == 't'
    assert args.format_real(sympy.Symbol('t') * 2 + 1) == '2*t + 1'