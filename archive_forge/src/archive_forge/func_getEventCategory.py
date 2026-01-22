import io, logging, socket, os, pickle, struct, time, re
from stat import ST_DEV, ST_INO, ST_MTIME
import queue
import threading
import copy
def getEventCategory(self, record):
    """
        Return the event category for the record.

        Override this if you want to specify your own categories. This version
        returns 0.
        """
    return 0