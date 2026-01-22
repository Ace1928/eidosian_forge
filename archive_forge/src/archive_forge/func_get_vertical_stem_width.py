from collections import namedtuple
import logging
import re
from ._mathtext_data import uni2type1
def get_vertical_stem_width(self):
    """
        Return the standard vertical stem width as float, or *None* if
        not specified in AFM file.
        """
    return self._header.get(b'StdVW', None)