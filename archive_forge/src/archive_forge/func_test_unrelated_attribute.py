import abc
from typing import NoReturn, Optional
import pytest
from cirq import ABCMetaImplementAnyOneOf, alternative
def test_unrelated_attribute():

    class _(metaclass=ABCMetaImplementAnyOneOf):
        _none_attribute = None
        _false_attribute = False
        _true_attribute = True

        @alternative(requires='alt', implementation=lambda self: None)
        def my_method(self):
            """my_method doc."""

        def alt(self):
            """alt doc."""