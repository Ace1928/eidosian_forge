import fixtures
from openstack import config
from openstack.config import cloud_region
from openstack import exceptions
from openstack.tests.unit.config import base
def test_environ_exists(self):
    c = config.OpenStackConfig(config_files=[self.cloud_yaml], vendor_files=[self.vendor_yaml], secure_files=[self.secure_yaml])
    cc = c.get_one('envvars')
    self._assert_cloud_details(cc)
    self.assertNotIn('auth_url', cc.config)
    self.assertIn('auth_url', cc.config['auth'])
    self.assertNotIn('project_id', cc.config['auth'])
    self.assertNotIn('auth_url', cc.config)
    cc = c.get_one('_test-cloud_')
    self._assert_cloud_details(cc)
    cc = c.get_one('_test_cloud_no_vendor')
    self._assert_cloud_details(cc)