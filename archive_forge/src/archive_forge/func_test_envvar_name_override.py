import fixtures
from openstack import config
from openstack.config import cloud_region
from openstack import exceptions
from openstack.tests.unit.config import base
def test_envvar_name_override(self):
    self.useFixture(fixtures.EnvironmentVariable('OS_CLOUD_NAME', 'override'))
    c = config.OpenStackConfig(config_files=[self.cloud_yaml], vendor_files=[self.vendor_yaml])
    cc = c.get_one('override')
    self._assert_cloud_details(cc)