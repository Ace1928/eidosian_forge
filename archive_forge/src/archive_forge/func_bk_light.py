import sys
from winappdbg import win32
from winappdbg import compat
from winappdbg.util import StaticClass
import re
import time
import struct
import traceback
@classmethod
def bk_light(cls):
    """Make the current background color light."""
    wAttributes = cls._get_text_attributes()
    wAttributes |= win32.BACKGROUND_INTENSITY
    cls._set_text_attributes(wAttributes)