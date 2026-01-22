import inspect
import time
import types
import unittest
from unittest.mock import (
from datetime import datetime
from functools import partial
def test_autospec_getattr_partial_function(self):

    class Foo:

        def __getattr__(self, attribute):
            return partial(lambda name: name, attribute)
    proxy = Foo()
    autospec = create_autospec(proxy)
    self.assertFalse(hasattr(autospec, '__name__'))