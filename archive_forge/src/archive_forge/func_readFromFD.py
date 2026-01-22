import errno
import os
from twisted.internet.main import CONNECTION_DONE, CONNECTION_LOST
def readFromFD(fd, callback):
    """
    Read from file descriptor, calling callback with resulting data.

    If successful, call 'callback' with a single argument: the
    resulting data.

    Returns same thing FileDescriptor.doRead would: CONNECTION_LOST,
    CONNECTION_DONE, or None.

    @type fd: C{int}
    @param fd: non-blocking file descriptor to be read from.
    @param callback: a callable which accepts a single argument. If
    data is read from the file descriptor it will be called with this
    data. Handling exceptions from calling the callback is up to the
    caller.

    Note that if the descriptor is still connected but no data is read,
    None will be returned but callback will not be called.

    @return: CONNECTION_LOST on error, CONNECTION_DONE when fd is
    closed, otherwise None.
    """
    try:
        output = os.read(fd, 8192)
    except OSError as ioe:
        if ioe.args[0] in (errno.EAGAIN, errno.EINTR):
            return
        else:
            return CONNECTION_LOST
    if not output:
        return CONNECTION_DONE
    callback(output)