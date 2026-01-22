from typing import (TYPE_CHECKING, Dict, List, Optional, TextIO, Tuple, Union,
from .lazy_import import lazy_import
from breezy import (
from breezy.bzr import (
import contextlib
import itertools
from . import config as _mod_config
from . import debug, errors, registry, repository
from . import revision as _mod_revision
from . import urlutils
from .controldir import (ControlComponent, ControlComponentFormat,
from .hooks import Hooks
from .inter import InterObject
from .lock import LogicalLockResult
from .revision import RevisionID
from .trace import is_quiet, mutter, mutter_callsite, note, warning
from .transport import Transport, get_transport
def dotted_revno_to_revision_id(self, revno, _cache_reverse=False):
    """Return the revision_id for a dotted revno.

        Args:
          revno: a tuple like (1,) or (1,1,2)
          _cache_reverse: a private parameter enabling storage
           of the reverse mapping in a top level cache. (This should
           only be done in selective circumstances as we want to
           avoid having the mapping cached multiple times.)
        Returns: the revision_id
        :raises errors.NoSuchRevision: if the revno doesn't exist
        """
    with self.lock_read():
        rev_id = self._do_dotted_revno_to_revision_id(revno)
        if _cache_reverse:
            self._partial_revision_id_to_revno_cache[rev_id] = revno
        return rev_id