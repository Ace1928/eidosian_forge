import io
import logging
import os
import pathlib
import shutil
import sys
import tempfile
from collections import OrderedDict
from contextlib import contextmanager
from typing import (IO, Dict, Iterable, Iterator, Mapping, Optional, Tuple,
from .parser import Binding, parse_stream
from .variables import parse_variables
def find_dotenv(filename: str='.env', raise_error_if_not_found: bool=False, usecwd: bool=False) -> str:
    """
    Search in increasingly higher folders for the given file

    Returns path to the file if found, or an empty string otherwise
    """

    def _is_interactive():
        """ Decide whether this is running in a REPL or IPython notebook """
        try:
            main = __import__('__main__', None, None, fromlist=['__file__'])
        except ModuleNotFoundError:
            return False
        return not hasattr(main, '__file__')
    if usecwd or _is_interactive() or getattr(sys, 'frozen', False):
        path = os.getcwd()
    else:
        frame = sys._getframe()
        current_file = __file__
        while frame.f_code.co_filename == current_file or not os.path.exists(frame.f_code.co_filename):
            assert frame.f_back is not None
            frame = frame.f_back
        frame_filename = frame.f_code.co_filename
        path = os.path.dirname(os.path.abspath(frame_filename))
    for dirname in _walk_to_root(path):
        check_path = os.path.join(dirname, filename)
        if os.path.isfile(check_path):
            return check_path
    if raise_error_if_not_found:
        raise IOError('File not found')
    return ''