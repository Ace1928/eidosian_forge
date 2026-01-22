from breezy import branch as _mod_branch
from breezy import check, controldir, errors
from breezy.revision import NULL_REVISION
from breezy.tests import TestNotApplicable, fixtures, transport_util
from breezy.tests.per_branch import TestCaseWithBranch
def test_clone_stacking_policy_handling(self):
    """Obey policy where possible, ignore otherwise."""
    if self.bzrdir_format.fixed_components:
        raise TestNotApplicable('Branch format 4 does not autoupgrade.')
    source = self.make_branch('source')
    stack_on = self.make_stacked_on_matching(source)
    parent_bzrdir = self.make_controldir('.', format='default')
    parent_bzrdir.get_config().set_default_stack_on('stack-on')
    target = source.controldir.clone('target').open_branch()
    if stack_on._format.supports_stacking():
        self.assertEqual('../stack-on', target.get_stacked_on_url())
    else:
        self.assertRaises(_mod_branch.UnstackableBranchFormat, target.get_stacked_on_url)