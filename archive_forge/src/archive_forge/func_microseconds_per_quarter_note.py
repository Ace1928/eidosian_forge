from __future__ import absolute_import, division, print_function
import sys
import math
import struct
import numpy as np
import warnings
@microseconds_per_quarter_note.setter
def microseconds_per_quarter_note(self, microseconds):
    """
        Set microseconds per quarter note.

        Parameters
        ----------
        microseconds : int
            Microseconds per quarter note.

        """
    self.data = [microseconds >> 16 - 8 * x & 255 for x in range(3)]