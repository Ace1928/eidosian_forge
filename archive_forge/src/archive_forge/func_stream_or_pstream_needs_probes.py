import errno
import os
import socket
import sys
import ovs.poller
import ovs.socket_util
import ovs.vlog
def stream_or_pstream_needs_probes(name):
    """ True if the stream or pstream specified by 'name' needs periodic probes
    to verify connectivity.  For [p]streams which need probes, it can take a
    long time to notice the connection was dropped.  Returns False if probes
    aren't needed, and None if 'name' is invalid"""
    cls = Stream._find_method(name)
    if cls:
        return cls.needs_probes()
    elif PassiveStream.is_valid_name(name):
        return PassiveStream.needs_probes(name)
    else:
        return None