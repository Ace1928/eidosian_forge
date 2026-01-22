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
def prune_dirnames(dirpath, dirnames):
    for i in range(len(dirnames) - 1, -1, -1):
        path = os.path.join(dirpath, dirnames[i])
        ip = os.path.join(os.path.relpath(path, basepath), '')
        if ignore_manager.is_ignored(ip):
            if not exclude_ignored:
                ignored_dirs.append(os.path.join(os.path.relpath(path, frompath), ''))
            del dirnames[i]
    return dirnames