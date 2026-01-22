from typing import List, Type, TYPE_CHECKING, Optional, Iterable
from .lazy_import import lazy_import
import time
from breezy import (
from breezy.i18n import gettext
from . import controldir, debug, errors, graph, registry, revision as _mod_revision, ui
from .decorators import only_raises
from .inter import InterObject
from .lock import LogicalLockResult, _RelockDebugMixin
from .revisiontree import RevisionTree
from .trace import (log_exception_quietly, mutter, mutter_callsite, note,
def verify_revision_signatures(self, revision_ids, gpg_strategy):
    """Verify revision signatures for a number of revisions.

        Args:
          revision_id: the revision to verify
          gpg_strategy: the GPGStrategy object to used

        Returns:
          Iterator over tuples with revision id, result and keys
        """
    with self.lock_read():
        for revid in revision_ids:
            result, key = self.verify_revision_signature(revid, gpg_strategy)
            yield (revid, result, key)