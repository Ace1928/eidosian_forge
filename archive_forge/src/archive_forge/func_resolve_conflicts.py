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
def resolve_conflicts(tt, pb=None, pass_func=None):
    """Make many conflict-resolution attempts, but die if they fail"""
    if pass_func is None:
        pass_func = conflict_pass
    new_conflicts = set()
    with ui.ui_factory.nested_progress_bar() as pb:
        for n in range(10):
            pb.update(gettext('Resolution pass'), n + 1, 10)
            conflicts = tt.find_raw_conflicts()
            if len(conflicts) == 0:
                return new_conflicts
            new_conflicts.update(pass_func(tt, conflicts))
        raise MalformedTransform(conflicts=conflicts)