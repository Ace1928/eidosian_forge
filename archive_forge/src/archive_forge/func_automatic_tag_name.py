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
def automatic_tag_name(self, revision_id):
    """Try to automatically find the tag name for a revision.

        Args:
          revision_id: Revision id of the revision.
        Returns: A tag name or None if no tag name could be determined.
        """
    for hook in Branch.hooks['automatic_tag_name']:
        ret = hook(self, revision_id)
        if ret is not None:
            return ret
    return None