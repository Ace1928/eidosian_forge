import base64
import binascii
import json
import time
import logging
from logging.config import dictConfig
from logging.config import fileConfig
import os
import socket
import sys
import threading
import traceback
from gunicorn import util
def parse_syslog_address(addr):
    if addr.startswith('unix://'):
        sock_type = None
        parts = addr.split('#', 1)
        if len(parts) == 2:
            addr = parts[0]
            if parts[1] == 'dgram':
                sock_type = socket.SOCK_DGRAM
        return (sock_type, addr.split('unix://')[1])
    if addr.startswith('udp://'):
        addr = addr.split('udp://')[1]
        socktype = socket.SOCK_DGRAM
    elif addr.startswith('tcp://'):
        addr = addr.split('tcp://')[1]
        socktype = socket.SOCK_STREAM
    else:
        raise RuntimeError('invalid syslog address')
    if '[' in addr and ']' in addr:
        host = addr.split(']')[0][1:].lower()
    elif ':' in addr:
        host = addr.split(':')[0].lower()
    elif addr == '':
        host = 'localhost'
    else:
        host = addr.lower()
    addr = addr.split(']')[-1]
    if ':' in addr:
        port = addr.split(':', 1)[1]
        if not port.isdigit():
            raise RuntimeError('%r is not a valid port number.' % port)
        port = int(port)
    else:
        port = 514
    return (socktype, (host, port))