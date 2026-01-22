from . import config as _mod_config
from . import errors, osutils
from .branch import Branch, BranchWriteLockResult
from .lock import LogicalLockResult, _RelockDebugMixin
from .revision import NULL_REVISION
from .tag import DisabledTags, MemoryTags
Get a breezy.config.BranchStack for this Branch.

        This can then be used to get and set configuration options for the
        branch.

        :return: A breezy.config.BranchStack.
        