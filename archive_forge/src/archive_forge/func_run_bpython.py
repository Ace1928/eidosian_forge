import fcntl
import os
import pty
import struct
import sys
import termios
import textwrap
import unittest
from bpython.test import TEST_CONFIG
from bpython.config import getpreferredencoding
def run_bpython(self, input):
    """
        Run bpython (with `backend` as backend) in a subprocess and
        enter the given input. Uses a test config that disables the
        paste detection.

        Returns bpython's output.
        """
    result = Deferred()
    encoding = getpreferredencoding()

    class Protocol(ProcessProtocol):
        STATES = SEND_INPUT, COLLECT = range(2)

        def __init__(self):
            self.data = ''
            self.delayed_call = None
            self.states = iter(self.STATES)
            self.state = next(self.states)

        def outReceived(self, data):
            self.data += data.decode(encoding)
            if self.delayed_call is not None:
                self.delayed_call.cancel()
            self.delayed_call = reactor.callLater(0.5, self.next)

        def next(self):
            self.delayed_call = None
            if self.state == self.SEND_INPUT:
                index = self.data.find('>>> ')
                if index >= 0:
                    self.data = self.data[index + 4:]
                    self.transport.write(input.encode(encoding))
                    self.state = next(self.states)
                elif self.data == '\x1b[6n':
                    self.transport.write('\x1b[2;1R'.encode(encoding))
            else:
                self.transport.closeStdin()
                if self.transport.pid is not None:
                    self.delayed_call = None
                    self.transport.signalProcess('TERM')

        def processExited(self, reason):
            if self.delayed_call is not None:
                self.delayed_call.cancel()
            result.callback(self.data)
    master, slave = pty.openpty()
    set_win_size(slave, 25, 80)
    reactor.spawnProcess(Protocol(), sys.executable, (sys.executable, '-m', f'bpython.{self.backend}', '--config', str(TEST_CONFIG), '-q'), env={'TERM': 'vt100', 'LANG': os.environ.get('LANG', 'C.UTF-8')}, usePTY=(master, slave, os.ttyname(slave)))
    return result