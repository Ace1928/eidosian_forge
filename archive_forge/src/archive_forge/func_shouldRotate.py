import glob
import os
import stat
import time
from typing import BinaryIO, Optional, cast
from twisted.python import threadable
def shouldRotate(self):
    """Rotate when the date has changed since last write"""
    return self.toDate() > self.lastDate