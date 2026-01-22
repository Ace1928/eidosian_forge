import os
import re
from typing import Iterator, List, Optional
from .errors import BzrError
def difference_index(atext, btext):
    """Find the indext of the first character that differs between two texts

    :param atext: The first text
    :type atext: str
    :param btext: The second text
    :type str: str
    :return: The index, or None if there are no differences within the range
    :rtype: int or NoneType
    """
    length = len(atext)
    if len(btext) < length:
        length = len(btext)
    for i in range(length):
        if atext[i] != btext[i]:
            return i
    return None