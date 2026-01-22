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
def makeunknown(self, tarinfo, targetpath):
    """Make a file from a TarInfo object with an unknown type
           at targetpath.
        """
    self.makefile(tarinfo, targetpath)
    self._dbg(1, 'tarfile: Unknown file type %r, extracted as regular file.' % tarinfo.type)