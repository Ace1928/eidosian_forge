import io, logging, socket, os, pickle, struct, time, re
from stat import ST_DEV, ST_INO, ST_MTIME
import queue
import threading
import copy
def createSocket(self):
    """
        Try to create a socket and, if it's not a datagram socket, connect it
        to the other end. This method is called during handler initialization,
        but it's not regarded as an error if the other end isn't listening yet
        --- the method will be called again when emitting an event,
        if there is no socket at that point.
        """
    address = self.address
    socktype = self.socktype
    if isinstance(address, str):
        self.unixsocket = True
        try:
            self._connect_unixsocket(address)
        except OSError:
            pass
    else:
        self.unixsocket = False
        if socktype is None:
            socktype = socket.SOCK_DGRAM
        host, port = address
        ress = socket.getaddrinfo(host, port, 0, socktype)
        if not ress:
            raise OSError('getaddrinfo returns an empty list')
        for res in ress:
            af, socktype, proto, _, sa = res
            err = sock = None
            try:
                sock = socket.socket(af, socktype, proto)
                if socktype == socket.SOCK_STREAM:
                    sock.connect(sa)
                break
            except OSError as exc:
                err = exc
                if sock is not None:
                    sock.close()
        if err is not None:
            raise err
        self.socket = sock
        self.socktype = socktype