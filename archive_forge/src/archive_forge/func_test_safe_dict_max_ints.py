from unittest import mock
from uuid import uuid4
import testtools
from openstack.cloud import _utils
from openstack import exceptions
from openstack.tests.unit import base
def test_safe_dict_max_ints(self):
    """Test integer comparison"""
    data = [{'f1': 3}, {'f1': 2}, {'f1': 1}]
    retval = _utils.safe_dict_max('f1', data)
    self.assertEqual(3, retval)