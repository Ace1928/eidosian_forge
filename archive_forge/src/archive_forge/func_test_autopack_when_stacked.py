from breezy import branch as _mod_branch
from breezy import check, controldir, errors
from breezy.revision import NULL_REVISION
from breezy.tests import TestNotApplicable, fixtures, transport_util
from breezy.tests.per_branch import TestCaseWithBranch
def test_autopack_when_stacked(self):
    stack_on = self.make_branch_and_tree('stack-on')
    if not stack_on.branch._format.supports_stacking():
        raise TestNotApplicable('%r does not support stacking' % self.branch_format)
    text_lines = [b'line %d blah blah blah\n' % i for i in range(20)]
    self.build_tree_contents([('stack-on/a', b''.join(text_lines))])
    stack_on.add('a')
    stack_on.commit('base commit')
    stacked_dir = stack_on.controldir.sprout('stacked', stacked=True)
    stacked_branch = stacked_dir.open_branch()
    local_tree = stack_on.controldir.sprout('local').open_workingtree()
    for i in range(20):
        text_lines[0] = b'changed in %d\n' % i
        self.build_tree_contents([('local/a', b''.join(text_lines))])
        local_tree.commit('commit %d' % i)
        local_tree.branch.push(stacked_branch)
    stacked_branch.repository.pack()
    check.check_dwim(stacked_branch.base, False, True, True)