import os
import random
import sys
import time
import xml.sax
import xml.sax.handler
from io import StringIO
from breezy import errors, osutils, trace, transport
from breezy.transport.http import urllib
def put_bytes(self, relpath, bytes, mode=None):
    """Copy the bytes object into the location.

        Tests revealed that contrary to what is said in
        http://www.rfc.net/rfc2068.html, the put is not
        atomic. When putting a file, if the client died, a
        partial file may still exists on the server.

        So we first put a temp file and then move it.

        :param relpath: Location to put the contents, relative to base.
        :param f:       File-like object.
        :param mode:    Not supported by DAV.
        """
    abspath = self._remote_path(relpath)
    stamp = '.tmp.%.9f.%d.%d' % (time.time(), os.getpid(), random.randint(0, 2147483647))
    tmp_relpath = relpath + stamp
    self.put_bytes_non_atomic(tmp_relpath, bytes)
    try:
        self.move(tmp_relpath, relpath)
    except Exception as e:
        exc_type, exc_val, exc_tb = sys.exc_info()
        try:
            self.delete(tmp_relpath)
        except:
            raise exc_type(exc_val).with_traceback(exc_tb)
        raise