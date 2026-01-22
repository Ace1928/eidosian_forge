from struct import pack, unpack, error as structError
from reportlab.lib.utils import bytestr, isUnicode, char2int, isStr, isBytes
from reportlab.pdfbase import pdfmetrics, pdfdoc
from reportlab import rl_config
from reportlab.lib.rl_accel import hex32, add32, calcChecksum, instanceStringWidthTTF
from collections import namedtuple
from io import BytesIO
import os, time
from reportlab.rl_config import register_reset
def seek_table(self, tag, offset_in_table=0):
    """Moves read pointer to the given offset within a given table and
        returns absolute offset of that position in the file."""
    self._pos = self.get_table_pos(tag)[0] + offset_in_table
    return self._pos