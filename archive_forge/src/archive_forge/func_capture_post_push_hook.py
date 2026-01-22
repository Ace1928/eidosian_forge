from io import BytesIO
from testtools.matchers import Equals, MatchesAny
from ... import branch, check, controldir, errors, push, tests
from ...branch import BindingUnsupported, Branch
from ...bzr import branch as bzrbranch
from ...bzr import vf_repository
from ...bzr.smart.repository import SmartServerRepositoryGetParentMap
from ...controldir import ControlDir
from ...revision import NULL_REVISION
from .. import test_server
from . import TestCaseWithInterBranch
def capture_post_push_hook(self, result):
    """Capture post push hook calls to self.hook_calls.

        The call is logged, as is some state of the two branches.
        """
    if result.local_branch:
        local_locked = result.local_branch.is_locked()
        local_base = result.local_branch.base
    else:
        local_locked = None
        local_base = None
    self.hook_calls.append(('post_push', result.source_branch, local_base, result.master_branch.base, result.old_revno, result.old_revid, result.new_revno, result.new_revid, result.source_branch.is_locked(), local_locked, result.master_branch.is_locked()))