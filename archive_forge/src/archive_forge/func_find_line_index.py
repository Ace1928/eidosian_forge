import argparse
import bisect
import os
import sys
import statistics
from typing import Dict, List, Set
from . import fitz
def find_line_index(values: List[int], value: int) -> int:
    """Find the right row coordinate.

        Args:
            values: (list) y-coordinates of rows.
            value: (int) lookup for this value (y-origin of char).
        Returns:
            y-ccordinate of appropriate line for value.
        """
    i = bisect.bisect_right(values, value)
    if i:
        return values[i - 1]
    raise RuntimeError('Line for %g not found in %s' % (value, values))