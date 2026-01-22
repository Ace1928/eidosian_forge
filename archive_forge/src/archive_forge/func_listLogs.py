import glob
import os
import stat
import time
from typing import BinaryIO, Optional, cast
from twisted.python import threadable
def listLogs(self):
    """
        Return sorted list of integers - the old logs' identifiers.
        """
    result = []
    for name in glob.glob('%s.*' % self.path):
        try:
            counter = int(name.split('.')[-1])
            if counter:
                result.append(counter)
        except ValueError:
            pass
    result.sort()
    return result