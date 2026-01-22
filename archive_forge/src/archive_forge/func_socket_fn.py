import sys, select
import pycurl
from io import BytesIO
def socket_fn(what, sock_fd, multi, socketp):
    if what == pycurl.POLL_IN or what == pycurl.POLL_INOUT:
        state['rlist'].append(sock_fd)
    elif what == pycurl.POLL_OUT or what == pycurl.POLL_INOUT:
        state['wlist'].append(sock_fd)
    elif what == pycurl.POLL_REMOVE:
        if sock_fd in state['rlist']:
            state['rlist'].remove(sock_fd)
        if sock_fd in state['wlist']:
            state['wlist'].remove(sock_fd)
    else:
        raise Exception('Unknown value of what: %s' % what)