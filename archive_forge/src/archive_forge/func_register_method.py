import errno
import os
import socket
import sys
import ovs.poller
import ovs.socket_util
import ovs.vlog
@staticmethod
def register_method(method, cls):
    Stream._SOCKET_METHODS[method + ':'] = cls