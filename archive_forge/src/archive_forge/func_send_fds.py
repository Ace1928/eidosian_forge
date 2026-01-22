fromfd() -- create a socket object from an open file descriptor [*]
fromshare() -- create a socket object from data received from socket.share() [*]
import _socket
from _socket import *
import os, sys, io, selectors
from enum import IntEnum, IntFlag
def send_fds(sock, buffers, fds, flags=0, address=None):
    """ send_fds(sock, buffers, fds[, flags[, address]]) -> integer

        Send the list of file descriptors fds over an AF_UNIX socket.
        """
    return sock.sendmsg(buffers, [(_socket.SOL_SOCKET, _socket.SCM_RIGHTS, array.array('i', fds))])