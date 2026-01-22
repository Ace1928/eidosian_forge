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
def preprocess_commit(self, revid, revobj, ref):
    if self.revid_to_mark.get(revid) or revid in self.excluded_revisions:
        return []
    if revobj is None:
        self.revid_to_mark[revid] = None
        return []
    if len(revobj.parent_ids) == 0:
        parent = breezy.revision.NULL_REVISION
    else:
        parent = revobj.parent_ids[0]
    self.revid_to_mark[revobj.revision_id] = b'%d' % (len(self.revid_to_mark) + 1)
    return [parent, revobj.revision_id]