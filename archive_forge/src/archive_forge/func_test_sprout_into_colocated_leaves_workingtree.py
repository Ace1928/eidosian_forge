from breezy import branchbuilder, errors, tests, urlutils
from breezy.branch import Branch
from breezy.controldir import BranchReferenceLoop
from breezy.tests import per_controldir
from breezy.tests.features import UnicodeFilenameFeature
def test_sprout_into_colocated_leaves_workingtree(self):
    if not self.bzrdir_format.is_supported():
        raise tests.TestNotApplicable('Control dir format not supported')
    if not self.bzrdir_format.supports_workingtrees:
        raise tests.TestNotApplicable('Control dir format does not support working trees')
    from_tree = self.make_branch_and_tree('from')
    self.build_tree_contents([('from/foo', 'contents')])
    from_tree.add(['foo'])
    revid1 = from_tree.commit('rev1')
    self.build_tree_contents([('from/foo', 'new contents')])
    revid2 = from_tree.commit('rev2')
    try:
        other_branch = self.make_branch_and_tree('to')
    except errors.UninitializableFormat:
        raise tests.TestNotApplicable('Control dir does not support creating new branches.')
    result = other_branch.controldir.push_branch(from_tree.branch, revision_id=revid1)
    self.assertTrue(result.workingtree_updated)
    self.assertFileEqual('contents', 'to/foo')
    from_tree.controldir.sprout(urlutils.join_segment_parameters(other_branch.user_url, {'branch': 'target'}), revision_id=revid2)
    active_branch = other_branch.controldir.open_branch(name='')
    self.assertEqual(revid1, active_branch.last_revision())
    to_branch = other_branch.controldir.open_branch(name='target')
    self.assertEqual(revid2, to_branch.last_revision())
    self.assertFileEqual('contents', 'to/foo')