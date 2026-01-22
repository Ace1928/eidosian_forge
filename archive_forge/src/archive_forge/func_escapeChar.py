from __future__ import absolute_import, division, unicode_literals
import re
import warnings
from .constants import DataLossWarning
def escapeChar(self, char):
    replacement = 'U%05X' % ord(char)
    self.replaceCache[char] = replacement
    return replacement