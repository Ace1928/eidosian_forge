import struct
import socket
import functools
import time
import logging
import Pyro4
def recv_message(sock):
    lengthbuf = recvall(sock, 4)
    if not lengthbuf:
        return None
    length, = struct.unpack('!I', lengthbuf)
    return recvall(sock, length)