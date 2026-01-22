import abc
from typing import NoReturn, Optional
import pytest
from cirq import ABCMetaImplementAnyOneOf, alternative
def test_doc_string():

    class SingleAlternative(metaclass=ABCMetaImplementAnyOneOf):

        def _default_impl(self, arg, kw=99):
            """Default implementation."""

        @alternative(requires='alt', implementation=_default_impl)
        def my_method(self, arg, kw=99) -> None:
            """my_method doc."""

        @abc.abstractmethod
        def alt(self) -> None:
            pass

    class SingleAlternativeChild(SingleAlternative):

        def alt(self) -> None:
            """Alternative method."""

    class SingleAlternativeOverride(SingleAlternative):

        def my_method(self, arg, kw=99) -> None:
            """my_method override."""

        def alt(self) -> None:
            """Unneeded alternative method."""
    assert SingleAlternative.my_method.__doc__ == 'my_method doc.'
    assert SingleAlternativeChild.my_method.__doc__ == 'my_method doc.'
    assert SingleAlternativeChild().my_method.__doc__ == 'my_method doc.'
    assert SingleAlternativeOverride.my_method.__doc__ == 'my_method override.'
    assert SingleAlternativeOverride().my_method.__doc__ == 'my_method override.'