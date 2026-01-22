from openstack import format as _format
from openstack import resource
from openstack.test import fakes
from openstack.tests.unit import base
def test_generate_fake_resource_one(self):
    res = fakes.generate_fake_resource(resource.Resource)
    self.assertIsInstance(res, resource.Resource)