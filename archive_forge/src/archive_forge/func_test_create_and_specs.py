import os
import sys
import six
import unittest2 as unittest
from mock.tests import support
from mock.tests.support import SomeClass, is_instance, callable
from mock import (
from mock.mock import _patch, _get_target
def test_create_and_specs(self):
    for kwarg in ('spec', 'spec_set', 'autospec'):
        p = patch('%s.doesnotexist' % __name__, create=True, **{kwarg: True})
        self.assertRaises(TypeError, p.start)
        self.assertRaises(NameError, lambda: doesnotexist)
        p = patch(MODNAME, create=True, **{kwarg: True})
        p.start()
        p.stop()