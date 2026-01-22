from unittest import mock
import uuid
import testtools
from openstack import connection
from openstack import exceptions
from openstack.tests.unit import base
from openstack import utils
def test_range_search_5(self):
    filters = {'key1': 'min', 'key2': 'min'}
    retval = self.cloud.range_search(RANGE_DATA, filters)
    self.assertIsInstance(retval, list)
    self.assertEqual(1, len(retval))
    self.assertEqual([RANGE_DATA[0]], retval)