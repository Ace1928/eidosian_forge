from unittest import mock
from keystoneauth1 import adapter
from openstack.common import tag
from openstack import exceptions
from openstack import resource
from openstack.tests.unit import base
from openstack.tests.unit.test_resource import FakeResponse
def test_tags_attribute(self):
    res = self.sot
    self.assertTrue(hasattr(res, 'tags'))
    self.assertIsInstance(res.tags, list)