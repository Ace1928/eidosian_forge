import binascii
import importlib.util
import io
import itertools
import os
import posixpath
import shutil
import stat
import struct
import sys
import threading
import time
import contextlib
import pathlib
def testzip(self):
    """Read all the files and check the CRC."""
    chunk_size = 2 ** 20
    for zinfo in self.filelist:
        try:
            with self.open(zinfo.filename, 'r') as f:
                while f.read(chunk_size):
                    pass
        except BadZipFile:
            return zinfo.filename