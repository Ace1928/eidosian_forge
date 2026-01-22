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
def start_background_thread(self, thread_name_suffix=''):
    self._started.clear()
    self._server_thread = threading.Thread(None, self.serve, args=(thread_name_suffix,), name='server-' + self.get_url(), daemon=True)
    self._server_thread.start()
    self._started.wait()