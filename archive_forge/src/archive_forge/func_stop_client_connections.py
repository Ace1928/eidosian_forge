import errno
import socket
import socketserver
import sys
import threading
from breezy import cethread, errors, osutils, transport, urlutils
from breezy.bzr.smart import medium, server
from breezy.transport import chroot, pathfilter
def stop_client_connections(self):
    while self.clients:
        c = self.clients.pop()
        self.shutdown_client(c)