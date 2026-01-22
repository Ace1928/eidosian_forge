import glob
import os
import stat
import time
from typing import BinaryIO, Optional, cast
from twisted.python import threadable
def toDate(self, *args):
    """Convert a unixtime to (year, month, day) localtime tuple,
        or return the current (year, month, day) localtime tuple.

        This function primarily exists so you may overload it with
        gmtime, or some cruft to make unit testing possible.
        """
    return time.localtime(*args)[:3]