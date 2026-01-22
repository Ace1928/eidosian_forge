from __future__ import absolute_import, division, unicode_literals
import re
import warnings
from .constants import DataLossWarning
def coerceComment(self, data):
    if self.preventDoubleDashComments:
        while '--' in data:
            warnings.warn('Comments cannot contain adjacent dashes', DataLossWarning)
            data = data.replace('--', '- -')
        if data.endswith('-'):
            warnings.warn('Comments cannot end in a dash', DataLossWarning)
            data += ' '
    return data