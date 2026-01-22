import yaml
from tempest.lib import exceptions
from heatclient.tests.functional.osc.v1 import base
def test_openstack_resource_type_show(self):
    rsrc_schema = self.openstack('orchestration resource type show OS::Heat::RandomString')
    self.assertIsInstance(yaml.safe_load(rsrc_schema), dict)