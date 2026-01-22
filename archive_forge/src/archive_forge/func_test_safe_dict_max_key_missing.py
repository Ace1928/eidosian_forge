from unittest import mock
from uuid import uuid4
import testtools
from openstack.cloud import _utils
from openstack import exceptions
from openstack.tests.unit import base
def test_safe_dict_max_key_missing(self):
    """Test missing key for an entry still works"""
    data = [{'f1': 3}, {'x': 2}, {'f1': 1}]
    retval = _utils.safe_dict_max('f1', data)
    self.assertEqual(3, retval)