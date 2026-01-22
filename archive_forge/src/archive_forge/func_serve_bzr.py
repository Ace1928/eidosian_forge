import errno
import os.path
import socket
import sys
import threading
import time
from ... import errors, trace
from ... import transport as _mod_transport
from ...hooks import Hooks
from ...i18n import gettext
from ...lazy_import import lazy_import
from breezy.bzr.smart import (
from breezy.transport import (
from breezy import (
def serve_bzr(transport, host=None, port=None, inet=False, timeout=None):
    """This is the default implementation of 'bzr serve'.

    It creates a TCP or pipe smart server on 'transport, and runs it.  The
    transport will be decorated with a chroot and pathfilter (using
    os.path.expanduser).
    """
    bzr_server = BzrServerFactory()
    try:
        bzr_server.set_up(transport, host, port, inet, timeout)
        bzr_server.smart_server.serve()
    except:
        hook_caught_exception = False
        for hook in SmartTCPServer.hooks['server_exception']:
            hook_caught_exception = hook(sys.exc_info())
        if not hook_caught_exception:
            raise
    finally:
        bzr_server.tear_down()