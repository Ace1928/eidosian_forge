from collections import namedtuple
import logging
import re
from ._mathtext_data import uni2type1
def get_capheight(self):
    """Return the cap height as float."""
    return self._header[b'CapHeight']