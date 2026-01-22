import sys
import threading
import traceback
import warnings
from _pydev_bundle._pydev_filesystem_encoding import getfilesystemencoding
from _pydev_bundle.pydev_imports import xmlrpclib, _queue
from _pydevd_bundle.pydevd_constants import Null
def force_server_kill():
    _ServerHolder.SERVER_COMM.notifications_queue.put_nowait(KillServer())