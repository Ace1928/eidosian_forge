import errno
import os
from twisted.internet.main import CONNECTION_DONE, CONNECTION_LOST
def setBlocking(fd):
    """
    Set the file description of the given file descriptor to blocking.
    """
    flags = fcntl.fcntl(fd, fcntl.F_GETFL)
    flags = flags & ~os.O_NONBLOCK
    fcntl.fcntl(fd, fcntl.F_SETFL, flags)