from __future__ import division
import sys
import unicodedata
from functools import reduce
def set_cols_width(self, array):
    """Set the desired columns width

        - the elements of the array should be integers, specifying the
          width of each column. For example:

                [10, 20, 5]
        """
    self._check_row_size(array)
    try:
        array = list(map(int, array))
        if reduce(min, array) <= 0:
            raise ValueError
    except ValueError:
        sys.stderr.write('Wrong argument in column width specification\n')
        raise
    self._width = array
    return self