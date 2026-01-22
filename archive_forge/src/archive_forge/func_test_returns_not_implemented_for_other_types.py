import fractions
import pytest
from cirq.testing.equals_tester import EqualsTester
def test_returns_not_implemented_for_other_types():

    class FirstClass:

        def __init__(self, val):
            self.val = val

        def __eq__(self, other):
            if not isinstance(other, FirstClass):
                return False
            return self.val == other.val

    class SecondClass:

        def __init__(self, val):
            self.val = val

        def __eq__(self, other):
            if isinstance(other, (FirstClass, SecondClass)):
                return self.val == other.val
            return NotImplemented
    assert SecondClass('a') == FirstClass('a')
    assert FirstClass('a') != SecondClass('a')

    class ThirdClass:

        def __init__(self, val):
            self.val = val

        def __eq__(self, other):
            if not isinstance(other, ThirdClass):
                return NotImplemented
            return self.val == other.val

    class FourthClass:

        def __init__(self, val):
            self.val = val

        def __eq__(self, other):
            if isinstance(other, (ThirdClass, FourthClass)):
                return self.val == other.val
            return NotImplemented
    assert ThirdClass('a') == FourthClass('a')
    assert FourthClass('a') == ThirdClass('a')
    eq = EqualsTester()
    with pytest.raises(AssertionError, match='NotImplemented'):
        eq.add_equality_group(FirstClass('a'), FirstClass('a'))
    eq = EqualsTester()
    eq.add_equality_group(ThirdClass('a'), ThirdClass('a'))