import sys
import time
from uuid import UUID
import pytest
from cherrypy._cpcompat import text_or_bytes
def markLog(self, key=None):
    """Insert a marker line into the log and set self.lastmarker."""
    if key is None:
        key = str(time.time())
    self.lastmarker = key
    with open(self.logfile, 'ab+') as f:
        f.write(b'%s%s\n' % (self.markerPrefix, key.encode('utf-8')))