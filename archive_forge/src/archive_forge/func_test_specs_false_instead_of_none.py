import os
import sys
import six
import unittest2 as unittest
from mock.tests import support
from mock.tests.support import SomeClass, is_instance, callable
from mock import (
from mock.mock import _patch, _get_target
def test_specs_false_instead_of_none(self):
    p = patch(MODNAME, spec=False, spec_set=False, autospec=False)
    mock = p.start()
    try:
        mock.does_not_exist
        mock.does_not_exist = 3
    finally:
        p.stop()