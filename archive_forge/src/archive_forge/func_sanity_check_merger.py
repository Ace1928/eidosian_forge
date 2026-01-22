import errno
import os
import sys
import breezy.bzr
import breezy.git
from . import controldir, errors, lazy_import, transport
import time
import breezy
from breezy import (
from breezy.branch import Branch
from breezy.transport import memory
from breezy.smtp_connection import SMTPConnection
from breezy.workingtree import WorkingTree
from breezy.i18n import gettext, ngettext
from .commands import Command, builtin_command_registry, display_command
from .option import (ListOption, Option, RegistryOption, _parse_revision_str,
from .revisionspec import RevisionInfo, RevisionSpec
from .trace import get_verbosity_level, is_quiet, mutter, note, warning
def sanity_check_merger(self, merger):
    if merger.show_base and merger.merge_type is not _mod_merge.Merge3Merger:
        raise errors.CommandError(gettext('Show-base is not supported for this merge type. %s') % merger.merge_type)
    if merger.reprocess is None:
        if merger.show_base:
            merger.reprocess = False
        else:
            merger.reprocess = merger.merge_type.supports_reprocess
    if merger.reprocess and (not merger.merge_type.supports_reprocess):
        raise errors.CommandError(gettext('Conflict reduction is not supported for merge type %s.') % merger.merge_type)
    if merger.reprocess and merger.show_base:
        raise errors.CommandError(gettext('Cannot do conflict reduction and show base.'))
    if merger.merge_type.requires_file_merge_plan and (not getattr(merger.this_tree, 'plan_file_merge', None) or not getattr(merger.other_tree, 'plan_file_merge', None) or (merger.base_tree is not None and (not getattr(merger.base_tree, 'plan_file_merge', None)))):
        raise errors.CommandError(gettext('Plan file merge unsupported: Merge type incompatible with tree formats.'))