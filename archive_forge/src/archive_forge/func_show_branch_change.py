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
def show_branch_change(branch, output, old_revno, old_revision_id):
    """Show the changes made to a branch.

    :param branch: The branch to show changes about.
    :param output: A file-like object to write changes to.
    :param old_revno: The revno of the old tip.
    :param old_revision_id: The revision_id of the old tip.
    """
    new_revno, new_revision_id = branch.last_revision_info()
    old_history, new_history = get_history_change(old_revision_id, new_revision_id, branch.repository)
    if old_history == [] and new_history == []:
        output.write('Nothing seems to have changed\n')
        return
    log_format = log_formatter_registry.get_default(branch)
    lf = log_format(show_ids=False, to_file=output, show_timezone='original')
    if old_history != []:
        output.write('*' * 60)
        output.write('\nRemoved Revisions:\n')
        show_flat_log(branch.repository, old_history, old_revno, lf)
        output.write('*' * 60)
        output.write('\n\n')
    if new_history != []:
        output.write('Added Revisions:\n')
        start_revno = new_revno - len(new_history) + 1
        show_log(branch, lf, verbose=False, direction='forward', start_revision=start_revno)