from ... import controldir as _mod_controldir
from ... import errors, lockable_files
from ...branch import BindingUnsupported, BranchFormat, BranchWriteLockResult
from ...bzr.fullhistory import FullHistoryBzrBranch
from ...decorators import only_raises
from ...lock import LogicalLockResult
from ...trace import mutter
def lock_read(self):
    """Lock the branch for read operations.

        :return: A breezy.lock.LogicalLockResult.
        """
    if not self.is_locked():
        self._note_lock('r')
    self.repository._warn_if_deprecated(self)
    self.repository.lock_read()
    try:
        self.control_files.lock_read()
        return LogicalLockResult(self.unlock)
    except:
        self.repository.unlock()
        raise