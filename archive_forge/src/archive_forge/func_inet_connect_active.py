import errno
import os
import os.path
import random
import socket
import sys
import ovs.fatal_signal
import ovs.poller
import ovs.vlog
def inet_connect_active(sock, address, family, dscp):
    try:
        set_nonblocking(sock)
        set_dscp(sock, family, dscp)
        error = sock.connect_ex(address)
        if error not in (0, errno.EINPROGRESS, errno.EWOULDBLOCK):
            sock.close()
            return error
        return 0
    except socket.error as e:
        sock.close()
        return get_exception_errno(e)