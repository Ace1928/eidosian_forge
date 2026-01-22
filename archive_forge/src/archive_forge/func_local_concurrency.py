import codecs
import errno
import os
import re
import stat
import sys
import time
from functools import partial
from typing import Dict, List
from .lazy_import import lazy_import
import locale
import ntpath
import posixpath
import select
import shutil
from shutil import rmtree
import socket
import subprocess
import unicodedata
from breezy import (
from breezy.i18n import gettext
from hashlib import md5
from hashlib import sha1 as sha
import breezy
from . import errors
def local_concurrency(use_cache=True):
    """Return how many processes can be run concurrently.

    Rely on platform specific implementations and default to 1 (one) if
    anything goes wrong.
    """
    global _cached_local_concurrency
    if _cached_local_concurrency is not None and use_cache:
        return _cached_local_concurrency
    concurrency = os.environ.get('BRZ_CONCURRENCY', None)
    if concurrency is None:
        import multiprocessing
        try:
            concurrency = multiprocessing.cpu_count()
        except NotImplementedError:
            try:
                concurrency = _local_concurrency()
            except OSError:
                pass
    try:
        concurrency = int(concurrency)
    except (TypeError, ValueError):
        concurrency = 1
    if use_cache:
        _cached_local_concurrency = concurrency
    return concurrency