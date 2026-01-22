from collections import namedtuple
import logging
import re
from ._mathtext_data import uni2type1
def get_width_char(self, c, isord=False):
    """
        Get the width of the character from the character metric WX field.
        """
    if not isord:
        c = ord(c)
    return self._metrics[c].width