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
def make_log_rev_iterator(branch, view_revisions, generate_delta, search, files=None, direction='reverse'):
    """Create a revision iterator for log.

    :param branch: The branch being logged.
    :param view_revisions: The revisions being viewed.
    :param generate_delta: Whether to generate a delta for each revision.
      Permitted values are None, 'full' and 'partial'.
    :param search: A user text search string.
    :param files: If non empty, only revisions matching one or more of
      the files are to be kept.
    :param direction: the direction in which view_revisions is sorted
    :return: An iterator over lists of ((rev_id, revno, merge_depth), rev,
        delta).
    """
    if isinstance(view_revisions, list):
        nones = [None] * len(view_revisions)
        log_rev_iterator = iter([list(zip(view_revisions, nones, nones))])
    else:

        def _convert():
            for view in view_revisions:
                yield (view, None, None)
        log_rev_iterator = iter([_convert()])
    for adapter in log_adapters:
        if adapter == _make_delta_filter:
            log_rev_iterator = adapter(branch, generate_delta, search, log_rev_iterator, files, direction)
        else:
            log_rev_iterator = adapter(branch, generate_delta, search, log_rev_iterator)
    return log_rev_iterator