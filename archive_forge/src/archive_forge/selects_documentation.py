import errno
import sys
from eventlet import patcher, support
from eventlet.hubs import hub
 Iterate through fds, removing the ones that are bad per the
        operating system.
        