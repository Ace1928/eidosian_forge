import sys
from winappdbg import win32
from winappdbg import compat
from winappdbg.util import StaticClass
import re
import time
import struct
import traceback
@classmethod
def bk_blue(cls):
    """Make the text background color blue."""
    wAttributes = cls._get_text_attributes()
    wAttributes &= ~win32.BACKGROUND_MASK
    wAttributes |= win32.BACKGROUND_BLUE
    cls._set_text_attributes(wAttributes)