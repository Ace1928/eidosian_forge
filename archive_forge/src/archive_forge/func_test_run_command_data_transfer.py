import socket
import threading
from io import StringIO
from unittest import skipIf
from dulwich.tests import TestCase
def test_run_command_data_transfer(self):
    vendor = ParamikoSSHVendor(allow_agent=False, look_for_keys=False)
    con = vendor.run_command('127.0.0.1', 'test_run_command_data_transfer', username=USER, port=self.port, password=PASSWORD)
    self.assertIn(b'test_run_command_data_transfer', self.commands)
    channel = self.transport.accept(5)
    channel.send(b'stdout\n')
    channel.send_stderr(b'stderr\n')
    channel.close()
    self.assertEqual(b'stdout\n', con.read(4096))