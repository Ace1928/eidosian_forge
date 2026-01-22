from unittest import mock
from uuid import uuid4
import testtools
from openstack.cloud import _utils
from openstack import exceptions
from openstack.tests.unit import base
def test_safe_dict_max_key_not_found(self):
    """Test key not found in any elements returns None"""
    data = [{'f1': 3}, {'f1': 2}, {'f1': 1}]
    retval = _utils.safe_dict_max('doesnotexist', data)
    self.assertIsNone(retval)