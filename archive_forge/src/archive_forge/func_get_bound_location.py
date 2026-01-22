from ... import controldir as _mod_controldir
from ... import errors, lockable_files
from ...branch import BindingUnsupported, BranchFormat, BranchWriteLockResult
from ...bzr.fullhistory import FullHistoryBzrBranch
from ...decorators import only_raises
from ...lock import LogicalLockResult
from ...trace import mutter
def get_bound_location(self):
    return None