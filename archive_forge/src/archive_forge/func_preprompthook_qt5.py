import os
import signal
import threading
from pydev_ipython.qt_for_kernel import QtCore, QtGui
from pydev_ipython.inputhook import allow_CTRL_C, ignore_CTRL_C, stdin_ready
def preprompthook_qt5(ishell):
    """'pre_prompt_hook' used to restore the Qt5 input hook

        (in case the latter was temporarily deactivated after a
        CTRL+C)
        """
    global got_kbdint, sigint_timer
    if sigint_timer:
        sigint_timer.cancel()
        sigint_timer = None
    if got_kbdint:
        mgr.set_inputhook(inputhook_qt5)
    got_kbdint = False