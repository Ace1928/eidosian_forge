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
def infolist(self):
    """Return a list of class ZipInfo instances for files in the
        archive."""
    return self.filelist