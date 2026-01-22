import errno
import os
import os.path
import random
import socket
import sys
import ovs.fatal_signal
import ovs.poller
import ovs.vlog
Returns an (error, bytes_written) tuple where 'error' is 0 on success,
    otherwise a positive errno value, and 'bytes_written' is the number of
    bytes that were written before the error occurred.  'error' is 0 if and
    only if 'bytes_written' is len(buf).