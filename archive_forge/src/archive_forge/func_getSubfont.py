from struct import pack, unpack, error as structError
from reportlab.lib.utils import bytestr, isUnicode, char2int, isStr, isBytes
from reportlab.pdfbase import pdfmetrics, pdfdoc
from reportlab import rl_config
from reportlab.lib.rl_accel import hex32, add32, calcChecksum, instanceStringWidthTTF
from collections import namedtuple
from io import BytesIO
import os, time
from reportlab.rl_config import register_reset
def getSubfont(self, subfontIndex):
    if self.fileKind != 'TTC':
        raise TTFError('"%s" is not a TTC file: use this method' % (self.filename, self.fileKind))
    try:
        pos = self.subfontOffsets[subfontIndex]
    except IndexError:
        raise TTFError('TTC file "%s": bad subfontIndex %s not in [0,%d]' % (self.filename, subfontIndex, self.numSubfonts - 1))
    self.seek(pos)
    self.readHeader()
    self.readTableDirectory()
    self.subfontNameX = bytestr('-' + str(subfontIndex))