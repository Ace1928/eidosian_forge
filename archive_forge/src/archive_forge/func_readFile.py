from struct import pack, unpack, error as structError
from reportlab.lib.utils import bytestr, isUnicode, char2int, isStr, isBytes
from reportlab.pdfbase import pdfmetrics, pdfdoc
from reportlab import rl_config
from reportlab.lib.rl_accel import hex32, add32, calcChecksum, instanceStringWidthTTF
from collections import namedtuple
from io import BytesIO
import os, time
from reportlab.rl_config import register_reset
def readFile(self, f):
    if not hasattr(self, '_ttf_data'):
        if hasattr(f, 'read'):
            self.filename = getattr(f, 'name', '(ttf)')
            self._ttf_data = f.read()
        else:
            self.filename, f = TTFOpenFile(f)
            self._ttf_data = f.read()
            f.close()
    self._pos = 0