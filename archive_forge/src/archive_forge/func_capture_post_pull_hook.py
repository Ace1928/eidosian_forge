from breezy import errors
from breezy.branch import BindingUnsupported, Branch
from breezy.controldir import ControlDir
from breezy.revision import NULL_REVISION
from breezy.tests import TestNotApplicable
from breezy.tests.per_interbranch import TestCaseWithInterBranch
def capture_post_pull_hook(self, result):
    """Capture post pull hook calls to self.hook_calls.

        The call is logged, as is some state of the two branches.
        """
    if result.local_branch:
        local_locked = result.local_branch.is_locked()
        local_base = result.local_branch.base
    else:
        local_locked = None
        local_base = None
    self.hook_calls.append(('post_pull', result.source_branch, local_base, result.master_branch.base, result.old_revno, result.old_revid, result.new_revno, result.new_revid, result.source_branch.is_locked(), local_locked, result.master_branch.is_locked()))