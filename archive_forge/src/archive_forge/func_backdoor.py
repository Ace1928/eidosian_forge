from code import InteractiveConsole
import errno
import socket
import sys
import eventlet
from eventlet import hubs
from eventlet.support import greenlets, get_errno
def backdoor(conn_info, locals=None):
    """Sets up an interactive console on a socket with a single connected
    client.  This does not block the caller, as it spawns a new greenlet to
    handle the console.  This is meant to be called from within an accept loop
    (such as backdoor_server).
    """
    conn, addr = conn_info
    if conn.family == socket.AF_INET:
        host, port = addr
        print('backdoor to %s:%s' % (host, port))
    elif conn.family == socket.AF_INET6:
        host, port, _, _ = addr
        print('backdoor to %s:%s' % (host, port))
    else:
        print('backdoor opened')
    fl = conn.makefile('rw')
    console = SocketConsole(fl, addr, locals)
    hub = hubs.get_hub()
    hub.schedule_call_global(0, console.switch)