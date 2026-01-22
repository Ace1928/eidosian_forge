import os
import sys
import six
import unittest2 as unittest
from mock.tests import support
from mock.tests.support import SomeClass, is_instance, callable
from mock import (
from mock.mock import _patch, _get_target
def test_patch_start_stop(self):
    original = something
    patcher = patch('%s.something' % __name__)
    self.assertIs(something, original)
    mock = patcher.start()
    try:
        self.assertIsNot(mock, original)
        self.assertIs(something, mock)
    finally:
        patcher.stop()
    self.assertIs(something, original)