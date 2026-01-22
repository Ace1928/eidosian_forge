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
def files_from_glob_patterns(patterns: List[str], access: int=os.F_OK) -> List[str]:
    """Return a list of file paths based on a list of glob patterns.

    Only files are returned, not directories, and optionally only files for which the user has a specified access to.

    :param patterns: list of file names and/or glob patterns
    :param access: file access type to verify (os.* where * is F_OK, R_OK, W_OK, or X_OK)
    :return: list of files matching the names and/or glob patterns
    """
    files = []
    for pattern in patterns:
        matches = files_from_glob_pattern(pattern, access=access)
        files.extend(matches)
    return files