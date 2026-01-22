from __future__ import annotations
import abc
import argparse
import os
import re
import typing as t
from .argcompletion import (
from .parsers import (
def detect_file_listing(value: str, mode: ParserMode) -> bool:
    """
    Return True if Bash will show a file listing and redraw the prompt, otherwise return False.

    If there are no list results, a file listing will be shown if the value after the last `=` or `:` character:

        - is empty
        - matches a full path
        - matches a partial path

    Otherwise Bash will play the bell sound and display nothing.

    see: https://github.com/kislyuk/argcomplete/issues/328
    see: https://github.com/kislyuk/argcomplete/pull/284
    """
    listing = False
    if mode == ParserMode.LIST:
        right = re.split('[=:]', value)[-1]
        listing = not right or os.path.exists(right)
        if not listing:
            directory = os.path.dirname(right)
            try:
                filenames = os.listdir(directory or '.')
            except Exception:
                pass
            else:
                listing = any((filename.startswith(right) for filename in filenames))
    return listing