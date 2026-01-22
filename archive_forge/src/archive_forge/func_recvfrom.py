from base64 import b64encode
import six
from errno import EOPNOTSUPP, EINVAL, EAGAIN
import functools
from io import BytesIO
import logging
import os
from os import SEEK_CUR
import socket
import struct
import sys
def recvfrom(self, bufsize, flags=0):
    if self.type != socket.SOCK_DGRAM:
        return super(socksocket, self).recvfrom(bufsize, flags)
    if not self._proxyconn:
        self.bind(('', 0))
    buf = BytesIO(super(socksocket, self).recv(bufsize + 1024, flags))
    buf.seek(2, SEEK_CUR)
    frag = buf.read(1)
    if ord(frag):
        raise NotImplementedError('Received UDP packet fragment')
    fromhost, fromport = self._read_SOCKS5_address(buf)
    if self.proxy_peername:
        peerhost, peerport = self.proxy_peername
        if fromhost != peerhost or peerport not in (0, fromport):
            raise socket.error(EAGAIN, 'Packet filtered')
    return (buf.read(bufsize), (fromhost, fromport))