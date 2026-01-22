from octavia_lib.api.drivers import exceptions
from octavia_lib.api.drivers import provider_base as driver_base
from octavia_lib.tests.unit import base
def test_get_supported_flavor_metadata(self):
    self.assertRaises(exceptions.NotImplementedError, self.driver.get_supported_flavor_metadata)