from __future__ import division
from __future__ import print_function
import collections
import contextlib
import errno
import functools
import os
import socket
import stat
import sys
import threading
import warnings
from collections import namedtuple
from socket import AF_INET
from socket import SOCK_DGRAM
from socket import SOCK_STREAM
def conn_to_ntuple(fd, fam, type_, laddr, raddr, status, status_map, pid=None):
    """Convert a raw connection tuple to a proper ntuple."""
    if fam in (socket.AF_INET, AF_INET6):
        if laddr:
            laddr = addr(*laddr)
        if raddr:
            raddr = addr(*raddr)
    if type_ == socket.SOCK_STREAM and fam in (AF_INET, AF_INET6):
        status = status_map.get(status, CONN_NONE)
    else:
        status = CONN_NONE
    fam = sockfam_to_enum(fam)
    type_ = socktype_to_enum(type_)
    if pid is None:
        return pconn(fd, fam, type_, laddr, raddr, status)
    else:
        return sconn(fd, fam, type_, laddr, raddr, status, pid)