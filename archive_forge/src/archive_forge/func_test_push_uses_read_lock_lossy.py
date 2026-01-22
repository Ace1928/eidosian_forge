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
def test_push_uses_read_lock_lossy(self):
    """Push should only need a read lock on the source side."""
    source = self.make_from_branch_and_tree('source')
    target = self.make_to_branch('target')
    self.build_tree(['source/a'])
    source.add(['a'])
    source.commit('a')
    try:
        with source.branch.lock_read(), target.lock_write():
            source.branch.push(target, stop_revision=source.last_revision(), lossy=True)
    except errors.LossyPushToSameVCS:
        raise tests.TestNotApplicable('push between branches of same format')