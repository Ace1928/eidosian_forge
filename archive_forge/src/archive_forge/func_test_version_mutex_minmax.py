import uuid
from oslo_config import cfg
from oslo_config import fixture as config
from keystoneauth1 import loading
from keystoneauth1.tests.unit.loading import utils
def test_version_mutex_minmax(self):
    self.conf_fx.config(service_type='type', service_name='name', valid_interfaces='iface', region_name='region', endpoint_override='endpoint', version='2.0', min_version='2.0', max_version='3.0', group=self.GROUP)
    self.assertRaises(TypeError, loading.load_adapter_from_conf_options, self.conf_fx.conf, self.GROUP, session='session', auth='auth')