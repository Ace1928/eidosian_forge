import sys
from winappdbg import win32
from winappdbg import compat
from winappdbg.util import StaticClass
import re
import time
import struct
import traceback
@classmethod
def bk_default(cls):
    """Make the current background color the default."""
    wAttributes = cls._get_text_attributes()
    wAttributes &= ~win32.BACKGROUND_MASK
    wAttributes &= ~win32.BACKGROUND_INTENSITY
    cls._set_text_attributes(wAttributes)