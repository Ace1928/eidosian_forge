import fractions
import pytest
from cirq.testing.equals_tester import EqualsTester
def test_works_on_types():
    eq = EqualsTester()
    eq.add_equality_group(object)
    eq.add_equality_group(int)
    eq.add_equality_group(object())