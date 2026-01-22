from struct import pack, unpack, error as structError
from reportlab.lib.utils import bytestr, isUnicode, char2int, isStr, isBytes
from reportlab.pdfbase import pdfmetrics, pdfdoc
from reportlab import rl_config
from reportlab.lib.rl_accel import hex32, add32, calcChecksum, instanceStringWidthTTF
from collections import namedtuple
from io import BytesIO
import os, time
from reportlab.rl_config import register_reset
def checksumFile(self):
    checksum = calcChecksum(self._ttf_data)
    if 2981146554 != checksum:
        raise TTFError('TTF file "%s": invalid checksum %s (expected 0xB1B0AFBA) len: %d &3: %d' % (self.filename, hex32(checksum), len(self._ttf_data), len(self._ttf_data) & 3))