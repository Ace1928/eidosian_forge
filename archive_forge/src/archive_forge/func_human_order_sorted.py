import os
import sys
import re
from collections.abc import Iterator
from warnings import warn
from looseversion import LooseVersion
import numpy as np
import textwrap
def human_order_sorted(l):
    """Sorts string in human order (i.e. 'stat10' will go after 'stat2')"""

    def atoi(text):
        return int(text) if text.isdigit() else text

    def natural_keys(text):
        if isinstance(text, tuple):
            text = text[0]
        return [atoi(c) for c in re.split('(\\d+)', text)]
    return sorted(l, key=natural_keys)