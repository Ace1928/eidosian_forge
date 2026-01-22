import os.path
import re
from contextlib import suppress
from typing import TYPE_CHECKING, BinaryIO, Dict, Iterable, List, Optional, Union
from .config import Config, get_xdg_config_home_path
def read_ignore_patterns(f: BinaryIO) -> Iterable[bytes]:
    """Read a git ignore file.

    Args:
      f: File-like object to read from
    Returns: List of patterns
    """
    for line in f:
        line = line.rstrip(b'\r\n')
        if not line.strip():
            continue
        if line.startswith(b'#'):
            continue
        while line.endswith(b' ') and (not line.endswith(b'\\ ')):
            line = line[:-1]
        line = line.replace(b'\\ ', b' ')
        yield line