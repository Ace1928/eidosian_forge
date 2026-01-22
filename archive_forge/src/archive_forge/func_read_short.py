from struct import pack, unpack, error as structError
from reportlab.lib.utils import bytestr, isUnicode, char2int, isStr, isBytes
from reportlab.pdfbase import pdfmetrics, pdfdoc
from reportlab import rl_config
from reportlab.lib.rl_accel import hex32, add32, calcChecksum, instanceStringWidthTTF
from collections import namedtuple
from io import BytesIO
import os, time
from reportlab.rl_config import register_reset
def read_short(self):
    """Reads a signed short"""
    self._pos += 2
    try:
        return unpack('>h', self._ttf_data[self._pos - 2:self._pos])[0]
    except structError as error:
        raise TTFError(error)