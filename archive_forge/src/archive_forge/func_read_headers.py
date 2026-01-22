import errno
import os
import socket
import sys
import six
from ._exceptions import *
from ._logging import *
from ._socket import*
from ._ssl_compat import *
from ._url import *
def read_headers(sock):
    status = None
    status_message = None
    headers = {}
    trace('--- response header ---')
    while True:
        line = recv_line(sock)
        line = line.decode('utf-8').strip()
        if not line:
            break
        trace(line)
        if not status:
            status_info = line.split(' ', 2)
            status = int(status_info[1])
            if len(status_info) > 2:
                status_message = status_info[2]
        else:
            kv = line.split(':', 1)
            if len(kv) == 2:
                key, value = kv
                headers[key.lower()] = value.strip()
            else:
                raise WebSocketException('Invalid header')
    trace('-----------------------')
    return (status, headers, status_message)