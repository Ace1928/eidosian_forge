import inspect
import time
import types
import unittest
from unittest.mock import (
from datetime import datetime
from functools import partial
def test_autospec_signature_classmethod(self):

    class Foo:

        @classmethod
        def class_method(cls, a, b=10, *, c):
            pass
    mock = create_autospec(Foo.__dict__['class_method'])
    self.assertEqual(inspect.signature(Foo.class_method), inspect.signature(mock))