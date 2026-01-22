import os
import sys
from collections import OrderedDict
import unittest
from unittest.test.testmock import support
from unittest.test.testmock.support import SomeClass, is_instance
from test.test_importlib.util import uncache
from unittest.mock import (
def test_autospec_classmethod(self):
    with patch('%s.Foo.class_method' % __name__, autospec=True) as method:
        Foo.class_method()
        method.assert_called_once_with()