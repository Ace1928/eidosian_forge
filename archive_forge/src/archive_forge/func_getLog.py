import glob
import os
import stat
import time
from typing import BinaryIO, Optional, cast
from twisted.python import threadable
def getLog(self, identifier):
    """Given a unix time, return a LogReader for an old log file."""
    if self.toDate(identifier) == self.lastDate:
        return self.getCurrentLog()
    filename = f'{self.path}.{self.suffix(identifier)}'
    if not os.path.exists(filename):
        raise ValueError('no such logfile exists')
    return LogReader(filename)