import os
import shutil
import tempfile
from twisted.internet.protocol import Protocol
from twisted.internet.test.test_serialport import DoNothing
from twisted.python.failure import Failure
from twisted.python.runtime import platform
from twisted.trial import unittest
def common_serialPortReturnsBytes(self, cbInQue):
    protocol = CollectReceivedProtocol()
    port = RegularFileSerialPort(protocol=protocol, deviceNameOrPortNumber=self.path, reactor=self.reactor, cbInQue=cbInQue)
    port.serialReadEvent()
    self.assertTrue(all((isinstance(d, bytes) for d in protocol.received_data)))
    port.connectionLost(Failure(Exception('Cleanup')))