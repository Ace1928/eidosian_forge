import codecs
import itertools
import re
import sys
from io import BytesIO
from typing import Callable, Dict, List
from warnings import warn
from .lazy_import import lazy_import
from breezy import (
from breezy.i18n import gettext, ngettext
from . import errors, registry
from . import revision as _mod_revision
from . import revisionspec, trace
from . import transport as _mod_transport
from .osutils import (format_date,
from .tree import InterTree, find_previous_path
def show_log(branch, lf, verbose=False, direction='reverse', start_revision=None, end_revision=None, limit=None, show_diff=False, match=None):
    """Write out human-readable log of commits to this branch.

    This function is being retained for backwards compatibility but
    should not be extended with new parameters. Use the new Logger class
    instead, eg. Logger(branch, rqst).show(lf), adding parameters to the
    make_log_request_dict function.

    :param lf: The LogFormatter object showing the output.

    :param verbose: If True show added/changed/deleted/renamed files.

    :param direction: 'reverse' (default) is latest to earliest; 'forward' is
        earliest to latest.

    :param start_revision: If not None, only show revisions >= start_revision

    :param end_revision: If not None, only show revisions <= end_revision

    :param limit: If set, shows only 'limit' revisions, all revisions are shown
        if None or 0.

    :param show_diff: If True, output a diff after each revision.

    :param match: Dictionary of search lists to use when matching revision
      properties.
    """
    if verbose:
        delta_type = 'full'
    else:
        delta_type = None
    if show_diff:
        diff_type = 'full'
    else:
        diff_type = None
    if isinstance(start_revision, int):
        try:
            start_revision = revisionspec.RevisionInfo(branch, start_revision)
        except (errors.NoSuchRevision, errors.RevnoOutOfBounds):
            raise errors.InvalidRevisionNumber(start_revision)
    if isinstance(end_revision, int):
        try:
            end_revision = revisionspec.RevisionInfo(branch, end_revision)
        except (errors.NoSuchRevision, errors.RevnoOutOfBounds):
            raise errors.InvalidRevisionNumber(end_revision)
    if end_revision is not None and end_revision.revno == 0:
        raise errors.InvalidRevisionNumber(end_revision.revno)
    rqst = make_log_request_dict(direction=direction, start_revision=start_revision, end_revision=end_revision, limit=limit, delta_type=delta_type, diff_type=diff_type)
    Logger(branch, rqst).show(lf)