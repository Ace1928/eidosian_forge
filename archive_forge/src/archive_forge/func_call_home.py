from ncclient import operations
from ncclient import transport
import socket
import logging
import functools
from ncclient.xml_ import *
def call_home(*args, **kwds):
    host = kwds['host']
    port = kwds.get('port', 4334)
    srv_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv_socket.bind((host, port))
    srv_socket.settimeout(10)
    srv_socket.listen()
    sock, remote_host = srv_socket.accept()
    logger.info('Callhome connection initiated from remote host {0}'.format(remote_host))
    kwds['sock'] = sock
    return connect_ssh(*args, **kwds)