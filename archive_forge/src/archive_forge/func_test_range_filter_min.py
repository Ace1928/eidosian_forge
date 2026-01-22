from unittest import mock
from uuid import uuid4
import testtools
from openstack.cloud import _utils
from openstack import exceptions
from openstack.tests.unit import base
def test_range_filter_min(self):
    retval = _utils.range_filter(RANGE_DATA, 'key1', 'min')
    self.assertIsInstance(retval, list)
    self.assertEqual(2, len(retval))
    self.assertEqual(RANGE_DATA[:2], retval)