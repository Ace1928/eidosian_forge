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
def show_flat_log(repository, history, last_revno, lf):
    """Show a simple log of the specified history.

    :param repository: The repository to retrieve revisions from.
    :param history: A list of revision_ids indicating the lefthand history.
    :param last_revno: The revno of the last revision_id in the history.
    :param lf: The log formatter to use.
    """
    revisions = repository.get_revisions(history)
    for i, rev in enumerate(revisions):
        lr = LogRevision(rev, i + last_revno, 0, None)
        lf.log_revision(lr)