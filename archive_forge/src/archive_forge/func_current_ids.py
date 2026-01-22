import os
import re
import signal as _signal
import sys
import time
import threading
import _thread
from cherrypy._cpcompat import text_or_bytes
from cherrypy._cpcompat import ntob
def current_ids():
    """Return the current (uid, gid) if available."""
    name, group = (None, None)
    if pwd:
        name = pwd.getpwuid(os.getuid())[0]
    if grp:
        group = grp.getgrgid(os.getgid())[0]
    return (name, group)