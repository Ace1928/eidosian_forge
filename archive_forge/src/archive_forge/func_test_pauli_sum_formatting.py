import collections
from typing import Union
import numpy as np
import pytest
import sympy
import sympy.parsing.sympy_parser as sympy_parser
import cirq
import cirq.testing
def test_pauli_sum_formatting():
    q = cirq.LineQubit.range(2)
    pauli = cirq.X(q[0])
    assert str(pauli) == 'X(q(0))'
    paulistr = cirq.X(q[0]) * cirq.X(q[1])
    assert str(paulistr) == 'X(q(0))*X(q(1))'
    paulisum1 = cirq.X(q[0]) * cirq.X(q[1]) + 4
    assert str(paulisum1) == '1.000*X(q(0))*X(q(1))+4.000*I'
    paulisum2 = cirq.X(q[0]) * cirq.X(q[1]) + cirq.Z(q[0])
    assert str(paulisum2) == '1.000*X(q(0))*X(q(1))+1.000*Z(q(0))'
    paulisum3 = cirq.X(q[0]) * cirq.X(q[1]) + cirq.Z(q[0]) * cirq.Z(q[1])
    assert str(paulisum3) == '1.000*X(q(0))*X(q(1))+1.000*Z(q(0))*Z(q(1))'
    assert f'{paulisum3:.0f}' == '1*X(q(0))*X(q(1))+1*Z(q(0))*Z(q(1))'
    empty = cirq.PauliSum.from_pauli_strings([])
    assert str(empty) == '0.000'