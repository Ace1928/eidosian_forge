import inspect
import time
import types
import unittest
from unittest.mock import (
from datetime import datetime
from functools import partial
def test_spec_has_descriptor_returning_function(self):

    class CrazyDescriptor(object):

        def __get__(self, obj, type_):
            if obj is None:
                return lambda x: None

    class MyClass(object):
        some_attr = CrazyDescriptor()
    mock = create_autospec(MyClass)
    mock.some_attr(1)
    with self.assertRaises(TypeError):
        mock.some_attr()
    with self.assertRaises(TypeError):
        mock.some_attr(1, 2)