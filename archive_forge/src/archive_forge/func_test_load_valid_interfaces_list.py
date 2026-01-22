import uuid
from oslo_config import cfg
from oslo_config import fixture as config
from keystoneauth1 import loading
from keystoneauth1.tests.unit.loading import utils
def test_load_valid_interfaces_list(self):
    self.conf_fx.config(service_type='type', service_name='name', valid_interfaces=['internal', 'public'], region_name='region', endpoint_override='endpoint', version='2.0', group=self.GROUP)
    adap = loading.load_adapter_from_conf_options(self.conf_fx.conf, self.GROUP, session='session', auth='auth')
    self.assertEqual('type', adap.service_type)
    self.assertEqual('name', adap.service_name)
    self.assertEqual(['internal', 'public'], adap.interface)
    self.assertEqual('region', adap.region_name)
    self.assertEqual('endpoint', adap.endpoint_override)
    self.assertEqual('session', adap.session)
    self.assertEqual('auth', adap.auth)
    self.assertEqual('2.0', adap.version)
    self.assertIsNone(adap.min_version)
    self.assertIsNone(adap.max_version)