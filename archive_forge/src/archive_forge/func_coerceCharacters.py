from __future__ import absolute_import, division, unicode_literals
import re
import warnings
from .constants import DataLossWarning
def coerceCharacters(self, data):
    if self.replaceFormFeedCharacters:
        for _ in range(data.count('\x0c')):
            warnings.warn('Text cannot contain U+000C', DataLossWarning)
        data = data.replace('\x0c', ' ')
    return data