import os
import platform
import shutil
import sys
import ctypes
from time import monotonic as clock
import configparser
from typing import Union
from .. import sparse
from .. import constants as const
import logging
import subprocess
from uuid import uuid4
def setTmpDir(self):
    """Set the tmpDir attribute to a reasonnable location for a temporary
        directory"""
    if os.name != 'nt':
        self.tmpDir = os.environ.get('TMPDIR', '/tmp')
        self.tmpDir = os.environ.get('TMP', self.tmpDir)
    else:
        self.tmpDir = os.environ.get('TMPDIR', '')
        self.tmpDir = os.environ.get('TMP', self.tmpDir)
        self.tmpDir = os.environ.get('TEMP', self.tmpDir)
    if not os.path.isdir(self.tmpDir):
        self.tmpDir = ''
    elif not os.access(self.tmpDir, os.F_OK + os.W_OK):
        self.tmpDir = ''