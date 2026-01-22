import sys
from twisted.internet import protocol, stdio
from twisted.python import reflect

Main program for the child process run by
L{twisted.test.test_stdio.StandardInputOutputTests.test_write} to test that
ITransport.write() works for process transports.
