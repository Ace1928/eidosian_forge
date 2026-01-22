from contextlib import ExitStack
import breezy.config
from . import debug, errors, trace, ui
from .branch import Branch
from .errors import BzrError, ConflictsInTree, StrictCommitFailed
from .i18n import gettext
from .osutils import (get_user_encoding, is_inside_any, minimum_path_selection,
from .trace import is_quiet, mutter, note
from .urlutils import unescape_for_display
def filter_excluded(iter_changes, exclude):
    """Filter exclude filenames.

    :param iter_changes: iter_changes function
    :param exclude: List of paths to exclude
    :return: iter_changes function
    """
    for change in iter_changes:
        new_excluded = change.path[1] is not None and is_inside_any(exclude, change.path[1])
        old_excluded = change.path[0] is not None and is_inside_any(exclude, change.path[0])
        if old_excluded and new_excluded:
            continue
        if old_excluded or new_excluded:
            continue
        yield change