from code import InteractiveConsole
import errno
import socket
import sys
import eventlet
from eventlet import hubs
from eventlet.support import greenlets, get_errno
def backdoor_server(sock, locals=None):
    """ Blocking function that runs a backdoor server on the socket *sock*,
    accepting connections and running backdoor consoles for each client that
    connects.

    The *locals* argument is a dictionary that will be included in the locals()
    of the interpreters.  It can be convenient to stick important application
    variables in here.
    """
    listening_on = sock.getsockname()
    if sock.family == socket.AF_INET:
        listening_on = '%s:%s' % listening_on
    elif sock.family == socket.AF_INET6:
        ip, port, _, _ = listening_on
        listening_on = '%s:%s' % (ip, port)
    print('backdoor server listening on %s' % (listening_on,))
    try:
        while True:
            socketpair = None
            try:
                socketpair = sock.accept()
                backdoor(socketpair, locals)
            except OSError as e:
                if get_errno(e) != errno.EPIPE:
                    raise
            finally:
                if socketpair:
                    socketpair[0].close()
    finally:
        sock.close()