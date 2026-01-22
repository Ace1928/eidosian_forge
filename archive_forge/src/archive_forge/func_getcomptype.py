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
def getcomptype(self):
    if self.buf.startswith(b'\x1f\x8b\x08'):
        return 'gz'
    elif self.buf[0:3] == b'BZh' and self.buf[4:10] == b'1AY&SY':
        return 'bz2'
    elif self.buf.startswith((b']\x00\x00\x80', b'\xfd7zXZ')):
        return 'xz'
    else:
        return 'tar'