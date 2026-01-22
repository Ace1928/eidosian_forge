from typing import Iterator
import pytest
import sympy
import cirq
from cirq.study import sweeps
from cirq_google.study import DeviceParameter
from cirq_google.api import v2
def test_list_sweep_bad_expression():
    with pytest.raises(TypeError, match='formula'):
        _ = cirq.ListSweep([cirq.ParamResolver({sympy.Symbol('a') + sympy.Symbol('b'): 4.0})])