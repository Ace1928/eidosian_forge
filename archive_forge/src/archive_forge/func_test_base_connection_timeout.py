import sys
import unittest
from unittest.mock import Mock, patch
from libcloud.common.base import LibcloudConnection
from libcloud.common.openstack import OpenStackBaseConnection
def test_base_connection_timeout(self):
    self.connection.connect()
    self.assertEqual(self.connection.timeout, self.timeout)
    self.connection.conn_class.assert_called_with(host='127.0.0.1', secure=1, port=443, timeout=10)