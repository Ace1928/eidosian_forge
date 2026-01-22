import datetime
import fnmatch
import os
import posixpath
import stat
import sys
import time
from collections import namedtuple
from contextlib import closing, contextmanager
from io import BytesIO, RawIOBase
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from .archive import tar_stream
from .client import get_transport_and_path
from .config import Config, ConfigFile, StackedConfig, read_submodules
from .diff_tree import (
from .errors import SendPackError
from .file import ensure_dir_exists
from .graph import can_fast_forward
from .ignore import IgnoreFilterManager
from .index import (
from .object_store import iter_tree_contents, tree_lookup_path
from .objects import (
from .objectspec import (
from .pack import write_pack_from_container, write_pack_index
from .patch import write_tree_diff
from .protocol import ZERO_SHA, Protocol
from .refs import (
from .repo import BaseRepo, Repo
from .server import (
from .server import update_server_info as server_update_server_info
def print_commit(commit, decode, outstream=sys.stdout):
    """Write a human-readable commit log entry.

    Args:
      commit: A `Commit` object
      outstream: A stream file to write to
    """
    outstream.write('-' * 50 + '\n')
    outstream.write('commit: ' + commit.id.decode('ascii') + '\n')
    if len(commit.parents) > 1:
        outstream.write('merge: ' + '...'.join([c.decode('ascii') for c in commit.parents[1:]]) + '\n')
    outstream.write('Author: ' + decode(commit.author) + '\n')
    if commit.author != commit.committer:
        outstream.write('Committer: ' + decode(commit.committer) + '\n')
    time_tuple = time.gmtime(commit.author_time + commit.author_timezone)
    time_str = time.strftime('%a %b %d %Y %H:%M:%S', time_tuple)
    timezone_str = format_timezone(commit.author_timezone).decode('ascii')
    outstream.write('Date:   ' + time_str + ' ' + timezone_str + '\n')
    outstream.write('\n')
    outstream.write(decode(commit.message) + '\n')
    outstream.write('\n')