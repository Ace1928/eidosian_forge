from code import InteractiveConsole
import errno
import socket
import sys
import eventlet
from eventlet import hubs
from eventlet.support import greenlets, get_errno
def switch_out(self):
    sys.stdin, sys.stderr, sys.stdout = self.saved