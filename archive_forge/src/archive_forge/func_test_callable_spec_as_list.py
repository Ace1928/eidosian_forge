import os
import sys
import six
import unittest2 as unittest
from mock.tests import support
from mock.tests.support import SomeClass, is_instance, callable
from mock import (
from mock.mock import _patch, _get_target
def test_callable_spec_as_list(self):
    spec = ('__call__',)
    p = patch(MODNAME, spec=spec)
    m = p.start()
    try:
        self.assertTrue(callable(m))
    finally:
        p.stop()