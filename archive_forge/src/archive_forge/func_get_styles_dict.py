import argparse
import collections
import functools
import glob
import inspect
import itertools
import os
import re
import subprocess
import sys
import threading
import unicodedata
from enum import (
from typing import (
from . import (
from .argparse_custom import (
def get_styles_dict(text: str) -> Dict[int, str]:
    """
    Return an OrderedDict containing all ANSI style sequences found in a string

    The structure of the dictionary is:
        key: index where sequences begins
        value: ANSI style sequence found at index in text

    Keys are in ascending order

    :param text: text to search for style sequences
    """
    from . import ansi
    start = 0
    styles = collections.OrderedDict()
    while True:
        match = ansi.ANSI_STYLE_RE.search(text, start)
        if match is None:
            break
        styles[match.start()] = match.group()
        start += len(match.group())
    return styles