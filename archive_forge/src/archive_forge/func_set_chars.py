from __future__ import division
import sys
import unicodedata
from functools import reduce
def set_chars(self, array):
    """Set the characters used to draw lines between rows and columns

        - the array should contain 4 fields:

            [horizontal, vertical, corner, header]

        - default is set to:

            ['-', '|', '+', '=']
        """
    if len(array) != 4:
        raise ArraySizeError('array should contain 4 characters')
    array = [x[:1] for x in [str(s) for s in array]]
    self._char_horiz, self._char_vert, self._char_corner, self._char_header = array
    return self