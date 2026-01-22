from __future__ import division
import sys
import unicodedata
from functools import reduce
def set_cols_dtype(self, array):
    """Set the desired columns datatype for the cols.

        - the elements of the array should be either a callable or any of
          "a", "t", "f", "e", "i" or "b":

            * "a": automatic (try to use the most appropriate datatype)
            * "t": treat as text
            * "f": treat as float in decimal format
            * "e": treat as float in exponential format
            * "i": treat as int
            * "b": treat as boolean
            * a callable: should return formatted string for any value given

        - by default, automatic datatyping is used for each column
        """
    self._check_row_size(array)
    self._dtype = array
    return self