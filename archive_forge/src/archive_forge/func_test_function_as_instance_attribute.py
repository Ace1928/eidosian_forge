import six
import unittest2 as unittest
from mock import (
from mock.mock import _Call, _CallList
from datetime import datetime
def test_function_as_instance_attribute(self):
    obj = SomeClass()

    def f(a):
        pass
    obj.f = f
    mock = create_autospec(obj)
    mock.f('bing')
    mock.f.assert_called_with('bing')