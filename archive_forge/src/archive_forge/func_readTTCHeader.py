from struct import pack, unpack, error as structError
from reportlab.lib.utils import bytestr, isUnicode, char2int, isStr, isBytes
from reportlab.pdfbase import pdfmetrics, pdfdoc
from reportlab import rl_config
from reportlab.lib.rl_accel import hex32, add32, calcChecksum, instanceStringWidthTTF
from collections import namedtuple
from io import BytesIO
import os, time
from reportlab.rl_config import register_reset
def readTTCHeader(self):
    self.ttcVersion = self.read_ulong()
    self.fileKind = 'TTC'
    self.ttfVersions = self.ttfVersions[:-1]
    if self.ttcVersion not in self.ttcVersions:
        raise TTFError('"%s" is not a %s file: can\'t read version 0x%8.8x' % (self.filename, self.fileKind, self.ttcVersion))
    self.numSubfonts = self.read_ulong()
    self.subfontOffsets = []
    a = self.subfontOffsets.append
    for i in range(self.numSubfonts):
        a(self.read_ulong())