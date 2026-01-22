import inspect
import io
import os
import re
import sys
import ast
from itertools import chain
from urllib.request import Request, urlopen
from urllib.parse import urlencode
from pathlib import Path
from IPython.core.error import TryNext, StdinNotImplementedError, UsageError
from IPython.core.macro import Macro
from IPython.core.magic import Magics, magics_class, line_magic
from IPython.core.oinspect import find_file, find_source_lines
from IPython.core.release import version
from IPython.testing.skipdoctest import skip_doctest
from IPython.utils.contexts import preserve_keys
from IPython.utils.path import get_py_filename
from warnings import warn
from logging import error
from IPython.utils.text import get_text_list
def extract_code_ranges(ranges_str):
    """Turn a string of range for %%load into 2-tuples of (start, stop)
    ready to use as a slice of the content split by lines.

    Examples
    --------
    list(extract_input_ranges("5-10 2"))
    [(4, 10), (1, 2)]
    """
    for range_str in ranges_str.split():
        rmatch = range_re.match(range_str)
        if not rmatch:
            continue
        sep = rmatch.group('sep')
        start = rmatch.group('start')
        end = rmatch.group('end')
        if sep == '-':
            start = int(start) - 1 if start else None
            end = int(end) if end else None
        elif sep == ':':
            start = int(start) - 1 if start else None
            end = int(end) - 1 if end else None
        else:
            end = int(start)
            start = int(start) - 1
        yield (start, end)