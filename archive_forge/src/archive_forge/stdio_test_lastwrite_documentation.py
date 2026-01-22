import sys
from twisted.internet.protocol import Protocol
from twisted.internet.stdio import StandardIO
from twisted.python.reflect import namedAny

Main program for the child process run by
L{twisted.test.test_stdio.StandardInputOutputTests.test_lastWriteReceived}
to test that L{os.write} can be reliably used after
L{twisted.internet.stdio.StandardIO} has finished.
