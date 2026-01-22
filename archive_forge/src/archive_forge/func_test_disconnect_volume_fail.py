import os
import tempfile
from unittest import mock
from os_brick import exception
from os_brick.initiator.connectors import huawei
from os_brick.tests.initiator import test_connector
def test_disconnect_volume_fail(self):
    """Test the fail disconnect volume case."""
    self.connector.connect_volume(self.connection_properties)
    self.assertEqual(True, HuaweiStorHyperConnectorTestCase.attached)
    self.assertRaises(exception.BrickException, self.connector_fail.disconnect_volume, self.connection_properties, self.device_info)
    expected_commands = [self.fake_sdscli_file + ' -c attach -v volume-b2911673-863c-4380-a5f2-e1729eecfe3f', self.fake_sdscli_file + ' -c querydev -v volume-b2911673-863c-4380-a5f2-e1729eecfe3f', self.fake_sdscli_file + ' -c detach -v volume-b2911673-863c-4380-a5f2-e1729eecfe3f']
    self.assertEqual(expected_commands, self.cmds)