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
def test_only_allowed_methods_exist(self):
    for spec in (['something'], ('something',)):
        for arg in ('spec', 'spec_set'):
            mock = Mock(**{arg: spec})
            mock.something
            self.assertRaisesRegex(AttributeError, "Mock object has no attribute 'something_else'", getattr, mock, 'something_else')