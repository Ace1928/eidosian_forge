import time
import configobj
from fastimport import commands
from fastimport import errors as plugin_errors
from fastimport import processor
from fastimport.helpers import invert_dictset
from .... import debug, delta, errors, osutils, progress
from .... import revision as _mod_revision
from ....bzr.knitpack_repo import KnitPackRepository
from ....trace import mutter, note, warning
from .. import (branch_updater, cache_manager, helpers, idmapfile, marks_file,
def dump_stats(self):
    time_required = progress.str_tdelta(time.time() - self._start_time)
    rc = self._revision_count - self.skip_total
    bc = self._branch_count
    wtc = self._tree_count
    self.note('Imported %d %s, updating %d %s and %d %s in %s', rc, helpers.single_plural(rc, 'revision', 'revisions'), bc, helpers.single_plural(bc, 'branch', 'branches'), wtc, helpers.single_plural(wtc, 'tree', 'trees'), time_required)