import base64
from tests.compat import unittest, mock
from tests.unit import AWSMockServiceTestCase
from boto.ec2.connection import EC2Connection
def test_multiple_private_ip_addresses(self):
    self.set_http_response(status_code=200)
    api_response = self.service_connection.get_all_reservations()
    self.assertEqual(len(api_response), 1)
    instances = api_response[0].instances
    self.assertEqual(len(instances), 1)
    instance = instances[0]
    self.assertEqual(len(instance.interfaces), 1)
    interface = instance.interfaces[0]
    self.assertEqual(len(interface.private_ip_addresses), 3)
    addresses = interface.private_ip_addresses
    self.assertEqual(addresses[0].private_ip_address, '10.0.0.67')
    self.assertTrue(addresses[0].primary)
    self.assertEqual(addresses[1].private_ip_address, '10.0.0.54')
    self.assertFalse(addresses[1].primary)
    self.assertEqual(addresses[2].private_ip_address, '10.0.0.55')
    self.assertFalse(addresses[2].primary)