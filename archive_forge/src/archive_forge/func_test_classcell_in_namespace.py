import abc
from typing import NoReturn, Optional
import pytest
from cirq import ABCMetaImplementAnyOneOf, alternative
def test_classcell_in_namespace():
    """Tests a historical issue where super() triggers python to add
    `__classcell__` to the namespace passed to the metaclass __new__.
    """

    class _(metaclass=ABCMetaImplementAnyOneOf):

        def other_method(self):
            super()