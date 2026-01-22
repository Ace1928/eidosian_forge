import re
import sys
from ast import literal_eval
from functools import total_ordering
from typing import NamedTuple, Sequence, Union
def parse_version_string(version: str=None) -> PythonVersionInfo:
    """
    Checks for a valid version number (e.g. `3.8` or `3.10.1` or `3`) and
    returns a corresponding version info that is always two characters long in
    decimal.
    """
    if version is None:
        version = '%s.%s' % sys.version_info[:2]
    if not isinstance(version, str):
        raise TypeError('version must be a string like "3.8"')
    return _parse_version(version)