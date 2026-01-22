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
def get_remote_repo(repo: Repo, remote_location: Optional[Union[str, bytes]]=None) -> Tuple[Optional[str], str]:
    config = repo.get_config()
    if remote_location is None:
        remote_location = get_branch_remote(repo)
    if isinstance(remote_location, str):
        encoded_location = remote_location.encode()
    else:
        encoded_location = remote_location
    section = (b'remote', encoded_location)
    remote_name: Optional[str] = None
    if config.has_section(section):
        remote_name = encoded_location.decode()
        encoded_location = config.get(section, 'url')
    else:
        remote_name = None
    return (remote_name, encoded_location.decode())