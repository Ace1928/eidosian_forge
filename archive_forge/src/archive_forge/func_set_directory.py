import errno
import operator
import os
import random
import re
import shlex
import socket
import stat
import string
import struct
import sys
import textwrap
import time
import traceback
from functools import reduce
from os import path
from typing import Optional
from twisted.internet import protocol, reactor, task
from twisted.persisted import styles
from twisted.protocols import basic
from twisted.python import _textattributes, log, reflect
def set_directory(self, directory):
    """
        Set the directory where the downloaded file will be placed.

        May raise OSError if the supplied directory path is not suitable.

        @param directory: The directory where the file to be received will be
            placed.
        @type directory: L{bytes}
        """
    if not path.exists(directory):
        raise OSError(errno.ENOENT, 'You see no directory there.', directory)
    if not path.isdir(directory):
        raise OSError(errno.ENOTDIR, 'You cannot put a file into something which is not a directory.', directory)
    if not os.access(directory, os.X_OK | os.W_OK):
        raise OSError(errno.EACCES, 'This directory is too hard to write in to.', directory)
    self.destDir = directory