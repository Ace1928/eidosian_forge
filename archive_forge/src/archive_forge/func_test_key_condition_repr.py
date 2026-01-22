import re
import pytest
import sympy
import cirq
def test_key_condition_repr():
    cirq.testing.assert_equivalent_repr(init_key_condition)
    cirq.testing.assert_equivalent_repr(cirq.KeyCondition(key_a, index=-2))