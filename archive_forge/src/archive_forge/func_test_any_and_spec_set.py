import inspect
import time
import types
import unittest
from unittest.mock import (
from datetime import datetime
from functools import partial
def test_any_and_spec_set(self):

    class Foo:

        def __eq__(self, other):
            pass
    mock = Mock(spec=Foo)
    mock(Foo(), 1)
    mock.assert_has_calls([call(ANY, 1)])
    mock.assert_called_with(ANY, 1)
    mock.assert_any_call(ANY, 1)