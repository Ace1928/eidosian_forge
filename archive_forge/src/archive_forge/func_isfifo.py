from builtins import open as bltn_open
import sys
import os
import io
import shutil
import stat
import time
import struct
import copy
import re
import warnings
def isfifo(self):
    """Return True if it is a FIFO."""
    return self.type == FIFOTYPE