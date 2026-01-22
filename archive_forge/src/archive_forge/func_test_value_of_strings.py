import fractions
import numpy as np
import pytest
import sympy
import cirq
def test_value_of_strings():
    assert cirq.ParamResolver().value_of('x') == sympy.Symbol('x')