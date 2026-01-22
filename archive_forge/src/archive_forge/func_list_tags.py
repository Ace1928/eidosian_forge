from collections import namedtuple
from contextlib import (
from io import BytesIO, RawIOBase
import datetime
import os
from pathlib import Path
import posixpath
import shutil
import stat
import sys
import time
from typing import (
from dulwich.archive import (
from dulwich.client import (
from dulwich.config import (
from dulwich.diff_tree import (
from dulwich.errors import (
from dulwich.graph import (
from dulwich.ignore import IgnoreFilterManager
from dulwich.index import (
from dulwich.object_store import (
from dulwich.objects import (
from dulwich.objectspec import (
from dulwich.pack import (
from dulwich.patch import write_tree_diff
from dulwich.protocol import (
from dulwich.refs import (
from dulwich.repo import BaseRepo, Repo
from dulwich.server import (
def list_tags(*args, **kwargs):
    import warnings
    warnings.warn('list_tags has been deprecated in favour of tag_list.', DeprecationWarning)
    return tag_list(*args, **kwargs)