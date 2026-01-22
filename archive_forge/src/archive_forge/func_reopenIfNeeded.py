import io, logging, socket, os, pickle, struct, time, re
from stat import ST_DEV, ST_INO, ST_MTIME
import queue
import threading
import copy
def reopenIfNeeded(self):
    """
        Reopen log file if needed.

        Checks if the underlying file has changed, and if it
        has, close the old stream and reopen the file to get the
        current stream.
        """
    try:
        sres = os.stat(self.baseFilename)
    except FileNotFoundError:
        sres = None
    if not sres or sres[ST_DEV] != self.dev or sres[ST_INO] != self.ino:
        if self.stream is not None:
            self.stream.flush()
            self.stream.close()
            self.stream = None
            self.stream = self._open()
            self._statstream()