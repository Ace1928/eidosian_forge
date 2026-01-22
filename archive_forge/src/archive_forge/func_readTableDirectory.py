from struct import pack, unpack, error as structError
from reportlab.lib.utils import bytestr, isUnicode, char2int, isStr, isBytes
from reportlab.pdfbase import pdfmetrics, pdfdoc
from reportlab import rl_config
from reportlab.lib.rl_accel import hex32, add32, calcChecksum, instanceStringWidthTTF
from collections import namedtuple
from io import BytesIO
import os, time
from reportlab.rl_config import register_reset
def readTableDirectory(self):
    try:
        self.numTables = self.read_ushort()
        self.searchRange = self.read_ushort()
        self.entrySelector = self.read_ushort()
        self.rangeShift = self.read_ushort()
        self.table = {}
        self.tables = []
        for n in range(self.numTables):
            record = {}
            record['tag'] = self.read_tag()
            record['checksum'] = self.read_ulong()
            record['offset'] = self.read_ulong()
            record['length'] = self.read_ulong()
            self.tables.append(record)
            self.table[record['tag']] = record
    except:
        raise TTFError('Corrupt %s file "%s" cannot read Table Directory' % (self.fileKind, self.filename))
    if self.validate:
        self.checksumTables()