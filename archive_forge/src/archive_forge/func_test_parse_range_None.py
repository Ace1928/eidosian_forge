from unittest import mock
from uuid import uuid4
import testtools
from openstack.cloud import _utils
from openstack import exceptions
from openstack.tests.unit import base
def test_parse_range_None(self):
    self.assertIsNone(_utils.parse_range(None))