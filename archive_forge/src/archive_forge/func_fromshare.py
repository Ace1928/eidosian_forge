fromfd() -- create a socket object from an open file descriptor [*]
fromshare() -- create a socket object from data received from socket.share() [*]
import _socket
from _socket import *
import os, sys, io, selectors
from enum import IntEnum, IntFlag
def fromshare(info):
    """ fromshare(info) -> socket object

        Create a socket object from the bytes object returned by
        socket.share(pid).
        """
    return socket(0, 0, 0, info)