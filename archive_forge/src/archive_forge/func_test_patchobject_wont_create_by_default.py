import os
import sys
import six
import unittest2 as unittest
from mock.tests import support
from mock.tests.support import SomeClass, is_instance, callable
from mock import (
from mock.mock import _patch, _get_target
def test_patchobject_wont_create_by_default(self):
    try:

        @patch.object(SomeClass, 'ord', sentinel.Frooble)
        def test():
            self.fail('Patching non existent attributes should fail')
        test()
    except AttributeError:
        pass
    else:
        self.fail('Patching non existent attributes should fail')
    self.assertFalse(hasattr(SomeClass, 'ord'))