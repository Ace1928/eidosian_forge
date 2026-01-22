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
def test_push_with_default_stacking_does_not_create_broken_branch(self):
    """Pushing a new standalone branch works even when there's a default
        stacking policy at the destination.

        The new branch will preserve the repo format (even if it isn't the
        default for the branch), and will be stacked when the repo format
        allows (which means that the branch format isn't necessarly preserved).
        """
    if isinstance(self.branch_format_from, bzrbranch.BranchReferenceFormat):
        raise tests.TestSkipped("BranchBuilder can't make reference branches.")
    repo = self.make_repository('repo', shared=True, format='1.6')
    try:
        builder = self.make_from_branch_builder('repo/local')
    except errors.UninitializableFormat:
        raise tests.TestNotApplicable('BranchBuilder can not initialize some formats')
    builder.start_series()
    revid1 = builder.build_snapshot(None, [('add', ('', None, 'directory', '')), ('add', ('filename', None, 'file', b'content\n'))])
    revid2 = builder.build_snapshot([revid1], [])
    revid3 = builder.build_snapshot([revid2], [('modify', ('filename', b'new-content\n'))])
    builder.finish_series()
    trunk = builder.get_branch()
    trunk.controldir.sprout(self.get_url('trunk'), revision_id=revid1)
    self.make_controldir('.').get_config().set_default_stack_on('trunk')
    output = BytesIO()
    push._show_push_branch(trunk, revid2, self.get_url('remote'), output)
    remote_branch = Branch.open(self.get_url('remote'))
    trunk.push(remote_branch)
    check.check_dwim(remote_branch.base, False, True, True)