from __future__ import annotations
import os
import re
from monty.io import zopen
def clean_lines(string_list, remove_empty_lines=True):
    """Strips whitespace, carriage returns and empty lines from a list of strings.

    Args:
        string_list: List of strings
        remove_empty_lines: Set to True to skip lines which are empty after
            stripping.

    Returns:
        List of clean strings with no whitespaces.
    """
    for s in string_list:
        clean_s = s
        if '#' in s:
            ind = s.index('#')
            clean_s = s[:ind]
        clean_s = clean_s.strip()
        if not remove_empty_lines or clean_s != '':
            yield clean_s