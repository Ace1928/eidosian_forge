import sys
import unittest
from unittest.mock import Mock, patch
from libcloud.common.base import LibcloudConnection
from libcloud.common.openstack import OpenStackBaseConnection
@patch('libcloud.test.common.test_openstack.OpenStackBaseConnection.connect', Mock())
def test_connection_is_reused_when_details_dont_change(self):
    url = 'https://example.com'
    self.connection._set_up_connection_info(url=url)
    self.assertEqual(self.connection.connect.call_count, 1)
    for index in range(0, 10):
        self.connection._set_up_connection_info(url=url)
        self.assertEqual(self.connection.connect.call_count, 1)