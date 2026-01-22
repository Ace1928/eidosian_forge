from contextlib import contextmanager
from typing import NamedTuple
from functools import partial
from IPython.core.guarded_eval import (
from IPython.testing import decorators as dec
import pytest
def test_external_not_installed():
    """
    Because attribute check requires checking if object is not of allowed
    external type, this tests logic for absence of external module.
    """

    class Custom:

        def __init__(self):
            self.test = 1

        def __getattr__(self, key):
            return key
    with module_not_installed('pandas'):
        context = limited(x=Custom())
        with pytest.raises(GuardRejection):
            guarded_eval('x.test', context)