from unittest import mock
from uuid import uuid4
import testtools
from openstack.cloud import _utils
from openstack import exceptions
from openstack.tests.unit import base
def test_parse_range_int_only(self):
    retval = _utils.parse_range('1024')
    self.assertIsInstance(retval, tuple)
    self.assertIsNone(retval[0])
    self.assertEqual(1024, retval[1])