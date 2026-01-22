import uuid
from oslo_config import cfg
from oslo_config import fixture as config
from keystoneauth1 import loading
from keystoneauth1.tests.unit.loading import utils
def test_load_bad_valid_interfaces_value(self):
    self.conf_fx.config(service_type='type', service_name='name', valid_interfaces='bad', region_name='region', endpoint_override='endpoint', version='2.0', group=self.GROUP)
    self.assertRaises(TypeError, loading.load_adapter_from_conf_options, self.conf_fx.conf, self.GROUP, session='session', auth='auth')