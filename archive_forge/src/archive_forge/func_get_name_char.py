from collections import namedtuple
import logging
import re
from ._mathtext_data import uni2type1
def get_name_char(self, c, isord=False):
    """Get the name of the character, i.e., ';' is 'semicolon'."""
    if not isord:
        c = ord(c)
    return self._metrics[c].name