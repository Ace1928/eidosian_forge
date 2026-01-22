from breezy import branch as _mod_branch
from breezy import check, controldir, errors
from breezy.revision import NULL_REVISION
from breezy.tests import TestNotApplicable, fixtures, transport_util
from breezy.tests.per_branch import TestCaseWithBranch
def test_fetch_revisions_with_file_changes(self):
    src_tree = self.make_branch_and_tree('src')
    self.build_tree_contents([('src/a', b'content')])
    src_tree.add('a')
    src_tree.commit('first commit')
    src_tree.controldir.sprout('stacked-on')
    target = self.make_branch('target')
    try:
        target.set_stacked_on_url('../stacked-on')
    except unstackable_format_errors as e:
        raise TestNotApplicable('Format does not support stacking.')
    self.build_tree_contents([('src/a', b'new content')])
    rev2 = src_tree.commit('second commit')
    target.fetch(src_tree.branch)
    rtree = target.repository.revision_tree(rev2)
    rtree.lock_read()
    self.addCleanup(rtree.unlock)
    self.assertEqual(b'new content', rtree.get_file_text('a'))
    self.check_lines_added_or_present(target, rev2)