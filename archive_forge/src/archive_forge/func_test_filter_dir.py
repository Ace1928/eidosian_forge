import copy
import pickle
import sys
import tempfile
import six
import unittest2 as unittest
import mock
from mock import (
from mock.mock import _CallList
from mock.tests.support import (
def test_filter_dir(self):
    patcher = patch.object(mock, 'FILTER_DIR', False)
    patcher.start()
    try:
        attrs = set(dir(Mock()))
        type_attrs = set(dir(Mock))
        self.assertEqual(set(), type_attrs - attrs)
    finally:
        patcher.stop()