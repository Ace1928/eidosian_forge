import uuid
from openstack.placement.v1 import resource_provider as _resource_provider
from openstack.tests.functional import base
def test_resource_provider(self):
    resource_providers = list(self.operator_cloud.placement.resource_providers())
    self.assertIsInstance(resource_providers[0], _resource_provider.ResourceProvider)
    self.assertIn(self.resource_provider_name, {x.name for x in resource_providers})
    resource_provider = self.operator_cloud.placement.find_resource_provider(self.resource_provider.name)
    self.assertEqual(self.resource_provider_name, resource_provider.name)
    resource_provider = self.operator_cloud.placement.get_resource_provider(self.resource_provider.id)
    self.assertEqual(self.resource_provider_name, resource_provider.name)
    new_resource_provider_name = self.getUniqueString()
    resource_provider = self.operator_cloud.placement.update_resource_provider(self.resource_provider, name=new_resource_provider_name, generation=self.resource_provider.generation)
    self.assertIsInstance(resource_provider, _resource_provider.ResourceProvider)
    self.assertEqual(new_resource_provider_name, resource_provider.name)