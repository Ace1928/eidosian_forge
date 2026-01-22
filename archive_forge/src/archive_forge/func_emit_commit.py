import re
import sys
import time
from email.utils import parseaddr
import breezy.branch
import breezy.revision
from ... import (builtins, errors, lazy_import, lru_cache, osutils, progress,
from ... import transport as _mod_transport
from . import helpers, marks_file
from fastimport import commands
def emit_commit(self, revobj, ref, tree_old, tree_new):
    if tree_old.get_revision_id() == breezy.revision.NULL_REVISION:
        self.print_cmd(commands.ResetCommand(ref, None))
    file_cmds = self._get_filecommands(tree_old, tree_new)
    mark = self.revid_to_mark[revobj.revision_id]
    self.print_cmd(self._get_commit_command(ref, mark, revobj, file_cmds))
    ncommits = len(self.revid_to_mark)
    self.report_progress(ncommits)
    if self.checkpoint is not None and self.checkpoint > 0 and ncommits and (ncommits % self.checkpoint == 0):
        self.note('Exported %i commits - adding checkpoint to output' % ncommits)
        self._save_marks()
        self.print_cmd(commands.CheckpointCommand())