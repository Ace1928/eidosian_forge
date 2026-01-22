import socket
def getsockopt(self, *args, **kwargs):
    return self._sock.getsockopt(*args, **kwargs)