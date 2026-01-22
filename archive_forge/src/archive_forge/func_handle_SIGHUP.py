import os
import re
import signal as _signal
import sys
import time
import threading
import _thread
from cherrypy._cpcompat import text_or_bytes
from cherrypy._cpcompat import ntob
def handle_SIGHUP(self):
    """Restart if daemonized, else exit."""
    if self._is_daemonized():
        self.bus.log('SIGHUP caught while daemonized. Restarting.')
        self.bus.restart()
    else:
        self.bus.log('SIGHUP caught but not daemonized. Exiting.')
        self.bus.exit()