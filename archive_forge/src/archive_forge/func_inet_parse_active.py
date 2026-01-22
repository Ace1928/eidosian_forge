import errno
import os
import os.path
import random
import socket
import sys
import ovs.fatal_signal
import ovs.poller
import ovs.vlog
def inet_parse_active(target, default_port):
    address = target.split(':')
    if len(address) >= 2:
        host_name = ':'.join(address[0:-1]).lstrip('[').rstrip(']')
        port = int(address[-1])
    else:
        if default_port:
            port = default_port
        else:
            raise ValueError('%s: port number must be specified' % target)
        host_name = address[0]
    if not host_name:
        raise ValueError('%s: bad peer name format' % target)
    return (host_name, port)