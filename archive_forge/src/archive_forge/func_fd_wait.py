import errno
import os
import select
import socket
import sys
import ovs.timeval
import ovs.vlog
def fd_wait(self, fd, events):
    """Registers 'fd' as waiting for the specified 'events' (which should
        be select.POLLIN or select.POLLOUT or their bitwise-OR).  The following
        call to self.block() will wake up when 'fd' becomes ready for one or
        more of the requested events.

        The event registration is one-shot: only the following call to
        self.block() is affected.  The event will need to be re-registered
        after self.block() is called if it is to persist.

        'fd' may be an integer file descriptor or an object with a fileno()
        method that returns an integer file descriptor."""
    self.poll.register(fd, events)