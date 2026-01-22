import sys
import threading
import traceback
import warnings
from _pydev_bundle._pydev_filesystem_encoding import getfilesystemencoding
from _pydev_bundle.pydev_imports import xmlrpclib, _queue
from _pydevd_bundle.pydevd_constants import Null
def initialize_server(port, daemon=False):
    if _ServerHolder.SERVER is None:
        if port is not None:
            notifications_queue = Queue()
            _ServerHolder.SERVER = ServerFacade(notifications_queue)
            _ServerHolder.SERVER_COMM = ServerComm(notifications_queue, port, daemon)
            _ServerHolder.SERVER_COMM.start()
        else:
            _ServerHolder.SERVER = Null()
            _ServerHolder.SERVER_COMM = Null()
    try:
        _ServerHolder.SERVER.notifyConnected()
    except:
        traceback.print_exc()