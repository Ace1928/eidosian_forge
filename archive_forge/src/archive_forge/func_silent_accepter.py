import base64
import errno
import json
from multiprocessing import connection
from multiprocessing import managers
import socket
import struct
import weakref
from oslo_rootwrap import wrapper
def silent_accepter(self):
    try:
        old_accepter(self)
    except EOFError:
        pass