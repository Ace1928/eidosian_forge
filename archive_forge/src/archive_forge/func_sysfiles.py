import os
import re
import signal as _signal
import sys
import time
import threading
import _thread
from cherrypy._cpcompat import text_or_bytes
from cherrypy._cpcompat import ntob
def sysfiles(self):
    """Return a Set of sys.modules filenames to monitor."""
    search_mod_names = filter(re.compile(self.match).match, list(sys.modules.keys()))
    mods = map(sys.modules.get, search_mod_names)
    return set(filter(None, map(self._file_for_module, mods)))