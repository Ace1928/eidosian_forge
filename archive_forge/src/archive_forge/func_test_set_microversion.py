import sys
import unittest
from unittest.mock import Mock, patch
from libcloud.common.base import LibcloudConnection
from libcloud.common.openstack import OpenStackBaseConnection
def test_set_microversion(self):
    self.connection.service_type = 'compute'
    self.connection._ex_force_microversion = '2.67'
    headers = self.connection.add_default_headers({})
    self.assertEqual(headers['OpenStack-API-Version'], 'compute 2.67')
    self.connection.service_type = 'compute'
    self.connection._ex_force_microversion = 'volume 2.67'
    headers = self.connection.add_default_headers({})
    self.assertNotIn('OpenStack-API-Version', headers)
    self.connection.service_type = 'volume'
    self.connection._ex_force_microversion = 'volume 2.67'
    headers = self.connection.add_default_headers({})
    self.assertEqual(headers['OpenStack-API-Version'], 'volume 2.67')