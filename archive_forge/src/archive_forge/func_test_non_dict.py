import collections.abc
import copy
import math
from unittest import mock
import ddt
from oslotest import base as test_base
import testscenarios
from oslo_utils import strutils
from oslo_utils import units
def test_non_dict(self):
    expected = {'password': '***', 'foo': 'bar'}
    payload = TestMapping()
    self.assertEqual(expected, strutils.mask_dict_password(payload))