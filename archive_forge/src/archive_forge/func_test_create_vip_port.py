from octavia_lib.api.drivers import exceptions
from octavia_lib.api.drivers import provider_base as driver_base
from octavia_lib.tests.unit import base
def test_create_vip_port(self):
    self.assertRaises(exceptions.NotImplementedError, self.driver.create_vip_port, False, False, False, None)