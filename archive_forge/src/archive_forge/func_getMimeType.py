from xdg.IniFile import IniFile, is_ascii
import xdg.Locale
from xdg.Exceptions import ParsingError
from xdg.util import which
import os.path
import re
import warnings
def getMimeType(self):
    """deprecated, use getMimeTypes instead """
    return self.get('MimeType', list=True, type='regex')