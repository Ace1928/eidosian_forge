import abc
from typing import NoReturn, Optional
import pytest
from cirq import ABCMetaImplementAnyOneOf, alternative
def test_two_alternatives():

    class TwoAlternatives(metaclass=ABCMetaImplementAnyOneOf):

        def _default_impl1(self, arg, kw=99):
            return f'default1({arg}, {kw}) ' + self.alt1()

        def _default_impl2(self, arg, kw=99):
            return f'default2({arg}, {kw}) ' + self.alt2()

        @alternative(requires='alt1', implementation=_default_impl1)
        @alternative(requires='alt2', implementation=_default_impl2)
        def my_method(self, arg, kw=99) -> str:
            """Docstring."""
            raise NotImplementedError

        @abc.abstractmethod
        def alt1(self) -> Optional[str]:
            pass

        @abc.abstractmethod
        def alt2(self) -> Optional[str]:
            pass

    class TwoAlternativesChild(TwoAlternatives):

        def alt1(self) -> str:
            return 'alt1'

        def alt2(self) -> NoReturn:
            raise RuntimeError

    class TwoAlternativesOverride(TwoAlternatives):

        def my_method(self, arg, kw=99) -> str:
            return 'override'

        def alt1(self) -> NoReturn:
            raise RuntimeError

        def alt2(self) -> NoReturn:
            raise RuntimeError

    class TwoAlternativesForceSecond(TwoAlternatives):

        def _do_alt1_with_my_method(self):
            return 'reverse ' + self.my_method(0, kw=0)

        @alternative(requires='my_method', implementation=_do_alt1_with_my_method)
        def alt1(self):
            """alt1 doc."""

        def alt2(self):
            return 'alt2'
    with pytest.raises(TypeError, match='abstract'):
        TwoAlternatives()
    assert TwoAlternativesChild().my_method(1) == 'default1(1, 99) alt1'
    assert TwoAlternativesChild().my_method(2, kw=3) == 'default1(2, 3) alt1'
    assert TwoAlternativesOverride().my_method(4, kw=5) == 'override'
    assert TwoAlternativesForceSecond().my_method(6, kw=7) == 'default2(6, 7) alt2'
    assert TwoAlternativesForceSecond().alt1() == 'reverse default2(0, 0) alt2'