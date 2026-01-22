from oslo_config import cfg
from oslo_config import types
from oslo_messaging._drivers import common as drv_cmn
from oslo_messaging.tests import utils as test_utils
from oslo_messaging import transport
def test_invalid_value(self):
    group = 'oslo_messaging_rabbit'
    self.config(kombu_reconnect_delay=1.0, group=group)
    url = transport.TransportURL.parse(self.conf, 'rabbit:///?kombu_reconnect_delay=invalid_value')
    self.assertRaises(ValueError, drv_cmn.ConfigOptsProxy, self.conf, url, group)