from collections import namedtuple
import logging
import re
from ._mathtext_data import uni2type1
def get_bbox_char(self, c, isord=False):
    if not isord:
        c = ord(c)
    return self._metrics[c].bbox