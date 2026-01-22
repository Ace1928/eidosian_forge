import contextlib
import errno
import os
import time
from stat import S_IEXEC, S_ISREG
from typing import Callable
from . import config as _mod_config
from . import controldir, errors, lazy_import, lock, osutils, registry, trace
from breezy import (
from breezy.i18n import gettext
from .errors import BzrError, DuplicateKey, InternalBzrError
from .filters import ContentFilterContext, filtered_output_bytes
from .mutabletree import MutableTree
from .osutils import delete_any, file_kind, pathjoin, sha_file, splitpath
from .progress import ProgressPhase
from .transport import FileExists, NoSuchFile
from .tree import InterTree, find_previous_path
def conflict_pass(tt, conflicts, path_tree=None):
    """Resolve some classes of conflicts.

    :param tt: The transform to resolve conflicts in
    :param conflicts: The conflicts to resolve
    :param path_tree: A Tree to get supplemental paths from
    """
    new_conflicts = set()
    for conflict in conflicts:
        resolver = CONFLICT_RESOLVERS.get(conflict[0])
        if resolver is None:
            continue
        new_conflicts.update(resolver(tt, path_tree, *conflict))
    return new_conflicts