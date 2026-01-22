from abc import ABCMeta
import copyreg
import functools
import io
import os
import pickle
import socket
import sys
from . import context
def sendfds(sock, fds):
    """Send an array of fds over an AF_UNIX socket."""
    fds = array.array('i', fds)
    msg = bytes([len(fds) % 256])
    sock.sendmsg([msg], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, fds)])
    if ACKNOWLEDGE and sock.recv(1) != b'A':
        raise RuntimeError('did not receive acknowledgement of fd')