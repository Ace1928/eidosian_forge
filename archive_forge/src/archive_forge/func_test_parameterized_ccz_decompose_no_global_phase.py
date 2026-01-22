import itertools
import numpy as np
import pytest
import sympy
import cirq
def test_parameterized_ccz_decompose_no_global_phase():
    decomposed_ops = cirq.decompose(cirq.CCZ(*cirq.LineQubit.range(3)) ** sympy.Symbol('theta'))
    assert not any((isinstance(op.gate, cirq.GlobalPhaseGate) for op in decomposed_ops))