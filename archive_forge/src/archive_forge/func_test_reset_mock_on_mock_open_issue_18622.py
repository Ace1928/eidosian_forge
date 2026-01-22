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
def test_reset_mock_on_mock_open_issue_18622(self):
    a = mock.mock_open()
    a.reset_mock()