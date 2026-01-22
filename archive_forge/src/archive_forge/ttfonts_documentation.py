from struct import pack, unpack, error as structError
from reportlab.lib.utils import bytestr, isUnicode, char2int, isStr, isBytes
from reportlab.pdfbase import pdfmetrics, pdfdoc
from reportlab import rl_config
from reportlab.lib.rl_accel import hex32, add32, calcChecksum, instanceStringWidthTTF
from collections import namedtuple
from io import BytesIO
import os, time
from reportlab.rl_config import register_reset
Makes  one or more PDF objects to be added to the document.  The
        caller supplies the internal name to be used (typically F1, F2, ... in
        sequence).

        This method creates a number of Font and FontDescriptor objects.  Every
        FontDescriptor is a (no more than) 256 character subset of the original
        TrueType font.