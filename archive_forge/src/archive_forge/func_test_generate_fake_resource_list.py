from openstack import format as _format
from openstack import resource
from openstack.test import fakes
from openstack.tests.unit import base
def test_generate_fake_resource_list(self):
    res = list(fakes.generate_fake_resources(resource.Resource, 2))
    self.assertEqual(2, len(res))
    self.assertIsInstance(res[0], resource.Resource)