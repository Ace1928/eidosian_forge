from abc import abstractmethod
import contextlib
from functools import wraps
import getpass
import logging
import os
import os.path as osp
import pathlib
import platform
import re
import shutil
import stat
import subprocess
import sys
import time
from urllib.parse import urlsplit, urlunsplit
import warnings
from typing import (
from .types import (
from gitdb.util import (  # noqa: F401  # @IgnorePep8
def remove_password_if_present(cmdline: Sequence[str]) -> List[str]:
    """Parse any command line argument and if one of the elements is an URL with a
    username and/or password, replace them by stars (in-place).

    If nothing is found, this just returns the command line as-is.

    This should be used for every log line that print a command line, as well as
    exception messages.
    """
    new_cmdline = []
    for index, to_parse in enumerate(cmdline):
        new_cmdline.append(to_parse)
        try:
            url = urlsplit(to_parse)
            if url.password is None and url.username is None:
                continue
            if url.password is not None:
                url = url._replace(netloc=url.netloc.replace(url.password, '*****'))
            if url.username is not None:
                url = url._replace(netloc=url.netloc.replace(url.username, '*****'))
            new_cmdline[index] = urlunsplit(url)
        except ValueError:
            continue
    return new_cmdline