import socket
from contextlib import contextmanager
from .exceptions import TIMEOUT, EOF
from .spawnbase import SpawnBase
def isalive(self):
    """ Alive if the fileno is valid """
    return self.socket.fileno() >= 0