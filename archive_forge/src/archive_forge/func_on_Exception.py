import sys
import logging
from . import compileUi, loadUi
def on_Exception(self, e):
    """ Handle a generic exception. """
    if logging.getLogger(self.LOGGER_NAME).level == logging.DEBUG:
        import traceback
        traceback.print_exception(*sys.exc_info())
    else:
        from PyQt5 import QtCore
        sys.stderr.write('An unexpected error occurred.\nCheck that you are using the latest version of PyQt5 and send an error report to\nsupport@riverbankcomputing.com, including the following information:\n\n  * your version of PyQt (%s)\n  * the UI file that caused this error\n  * the debug output of pyuic5 (use the -d flag when calling pyuic5)\n' % QtCore.PYQT_VERSION_STR)