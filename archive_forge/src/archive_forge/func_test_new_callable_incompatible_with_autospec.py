import os
import sys
import six
import unittest2 as unittest
from mock.tests import support
from mock.tests.support import SomeClass, is_instance, callable
from mock import (
from mock.mock import _patch, _get_target
def test_new_callable_incompatible_with_autospec(self):
    self.assertRaises(ValueError, patch, foo_name, new_callable=MagicMock, autospec=True)
    self.assertRaises(ValueError, patch.object, Foo, 'f', new_callable=MagicMock, autospec=True)