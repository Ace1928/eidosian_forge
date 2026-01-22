import os
import shutil
import tempfile
from twisted.internet.protocol import Protocol
from twisted.internet.test.test_serialport import DoNothing
from twisted.python.failure import Failure
from twisted.python.runtime import platform
from twisted.trial import unittest
def common_exerciseHandleAccess(self, cbInQue):
    port = RegularFileSerialPort(protocol=self.protocol, deviceNameOrPortNumber=self.path, reactor=self.reactor, cbInQue=cbInQue)
    port.serialReadEvent()
    port.write(b'')
    port.write(b'abcd')
    port.write(b'ABCD')
    port.serialWriteEvent()
    port.serialWriteEvent()
    port.connectionLost(Failure(Exception('Cleanup')))