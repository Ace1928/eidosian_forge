import fixtures
from openstack import config
from openstack.config import cloud_region
from openstack import exceptions
from openstack.tests.unit.config import base
def test_get_one_with_config_files(self):
    c = config.OpenStackConfig(config_files=[self.cloud_yaml], vendor_files=[self.vendor_yaml], secure_files=[self.secure_yaml])
    self.assertIsInstance(c.cloud_config, dict)
    self.assertIn('cache', c.cloud_config)
    self.assertIsInstance(c.cloud_config['cache'], dict)
    self.assertIn('max_age', c.cloud_config['cache'])
    self.assertIn('path', c.cloud_config['cache'])
    cc = c.get_one('_test-cloud_')
    self._assert_cloud_details(cc)
    cc = c.get_one('_test_cloud_no_vendor')
    self._assert_cloud_details(cc)