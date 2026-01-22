import os
import sys
import six
import unittest2 as unittest
from mock.tests import support
from mock.tests.support import SomeClass, is_instance, callable
from mock import (
from mock.mock import _patch, _get_target
def test_spec_set_true(self):
    for kwarg in ('spec', 'autospec'):
        p = patch(MODNAME, spec_set=True, **{kwarg: True})
        m = p.start()
        try:
            self.assertRaises(AttributeError, setattr, m, 'doesnotexist', 'something')
            self.assertRaises(AttributeError, getattr, m, 'doesnotexist')
        finally:
            p.stop()