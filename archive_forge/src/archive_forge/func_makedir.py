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
def makedir(self, tarinfo, targetpath):
    """Make a directory called targetpath.
        """
    try:
        if tarinfo.mode is None:
            os.mkdir(targetpath)
        else:
            os.mkdir(targetpath, 448)
    except FileExistsError:
        if not os.path.isdir(targetpath):
            raise