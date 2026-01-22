from struct import pack, unpack, error as structError
from reportlab.lib.utils import bytestr, isUnicode, char2int, isStr, isBytes
from reportlab.pdfbase import pdfmetrics, pdfdoc
from reportlab import rl_config
from reportlab.lib.rl_accel import hex32, add32, calcChecksum, instanceStringWidthTTF
from collections import namedtuple
from io import BytesIO
import os, time
from reportlab.rl_config import register_reset
def makeStream(self):
    """Finishes the generation and returns the TTF file as a string"""
    stm = BytesIO()
    write = stm.write
    tables = self.tables
    numTables = len(tables)
    searchRange = 1
    entrySelector = 0
    while searchRange * 2 <= numTables:
        searchRange = searchRange * 2
        entrySelector = entrySelector + 1
    searchRange = searchRange * 16
    rangeShift = numTables * 16 - searchRange
    write(pack('>lHHHH', 65536, numTables, searchRange, entrySelector, rangeShift))
    offset = 12 + numTables * 16
    wStr = lambda x: write(bytes(tag, 'latin1'))
    tables_items = list(sorted(tables.items()))
    for tag, data in tables_items:
        if tag == 'head':
            head_start = offset
        checksum = calcChecksum(data)
        wStr(tag)
        write(pack('>LLL', checksum, offset, len(data)))
        paddedLength = len(data) + 3 & ~3
        offset = offset + paddedLength
    for tag, data in tables_items:
        data += b'\x00\x00\x00'
        write(data[:len(data) & ~3])
    checksum = calcChecksum(stm.getvalue())
    checksum = add32(2981146554, -checksum)
    stm.seek(head_start + 8)
    write(pack('>L', checksum))
    return stm.getvalue()