import os
import shutil
import tempfile
from twisted.internet.protocol import Protocol
from twisted.internet.test.test_serialport import DoNothing
from twisted.python.failure import Failure
from twisted.python.runtime import platform
from twisted.trial import unittest
def test_serialPortInitiallyConnected(self):
    """
        Test the port is connected at initialization time, and
        C{Protocol.makeConnection} has been called on the desired protocol.
        """
    self.assertEqual(0, self.protocol.connected)
    port = RegularFileSerialPort(self.protocol, self.path, self.reactor)
    self.assertEqual(1, port.connected)
    self.assertEqual(1, self.protocol.connected)
    self.assertEqual(port, self.protocol.transport)
    port.connectionLost(Failure(Exception('Cleanup')))