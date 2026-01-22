import atexit
import inspect
import multiprocessing.connection
import os
import signal
import subprocess
import sys
import time
import pickle
from ..Qt import QT_LIB, mkQApp
from ..util import cprint  # color printing for debugging
from .remoteproxy import (
import threading
def startQtEventLoop(name, port, authkey, ppid, debug=False):
    if debug:
        import os
        cprint.cout(debug, '[%d] connecting to server at port localhost:%d, authkey=%s..\n' % (os.getpid(), port, repr(authkey)), -1)
    conn = multiprocessing.connection.Client(('localhost', int(port)), authkey=authkey)
    if debug:
        cprint.cout(debug, '[%d] connected; starting remote proxy.\n' % os.getpid(), -1)
    from ..Qt import QtWidgets
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = mkQApp()
        app.setQuitOnLastWindowClosed(False)
    global HANDLER
    HANDLER = RemoteQtEventHandler(conn, name, ppid, debug=debug)
    HANDLER.startEventTimer()
    app.exec() if hasattr(app, 'exec') else app.exec_()