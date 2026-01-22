from __future__ import absolute_import, division, unicode_literals
import re
import warnings
from .constants import DataLossWarning
def getReplacementCharacter(self, char):
    if char in self.replaceCache:
        replacement = self.replaceCache[char]
    else:
        replacement = self.escapeChar(char)
    return replacement