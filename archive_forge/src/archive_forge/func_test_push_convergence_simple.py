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
def test_push_convergence_simple(self):
    mine = self.make_from_branch_and_tree('mine')
    mine.commit('1st post', allow_pointless=True)
    try:
        other = self.sprout_to(mine.controldir, 'other').open_workingtree()
    except errors.NoRoundtrippingSupport:
        raise tests.TestNotApplicable('lossless push between %r and %r not supported' % (self.branch_format_from, self.branch_format_to))
    m1 = other.commit('my change', allow_pointless=True)
    try:
        mine.merge_from_branch(other.branch)
    except errors.NoRoundtrippingSupport:
        raise tests.TestNotApplicable('lossless push between %r and %r not supported' % (self.branch_format_from, self.branch_format_to))
    p2 = mine.commit('merge my change')
    result = mine.branch.push(other.branch)
    self.assertEqual(p2, other.branch.last_revision())
    self.assertEqual(result.old_revid, m1)
    self.assertEqual(result.new_revid, p2)