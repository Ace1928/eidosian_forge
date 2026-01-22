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
def get_exes_in_path(starts_with: str) -> List[str]:
    """Returns names of executables in a user's path

    :param starts_with: what the exes should start with. leave blank for all exes in path.
    :return: a list of matching exe names
    """
    wildcards = ['*', '?']
    for wildcard in wildcards:
        if wildcard in starts_with:
            return []
    env_path = os.getenv('PATH')
    if env_path is None:
        paths = []
    else:
        paths = [p for p in env_path.split(os.path.pathsep) if not os.path.islink(p)]
    exes_set = set()
    for path in paths:
        full_path = os.path.join(path, starts_with)
        matches = files_from_glob_pattern(full_path + '*', access=os.X_OK)
        for match in matches:
            exes_set.add(os.path.basename(match))
    return list(exes_set)