from io import BytesIO
from ... import errors, tests, ui
from . import TestCaseWithBranch
def test_check_detects_invalid_revhistory(self):
    tree = self.make_branch_and_tree('test')
    r1 = tree.commit('one')
    r2 = tree.commit('two')
    r3 = tree.commit('three')
    r4 = tree.commit('four')
    tree.set_parent_ids([r1])
    tree.branch.set_last_revision_info(1, r1)
    r2b = tree.commit('two-b')
    tree.set_parent_ids([r4, r2b])
    tree.branch.set_last_revision_info(4, r4)
    r5 = tree.commit('five')
    if getattr(tree.branch, '_set_revision_history', None) is not None:
        tree.branch._set_revision_history([r1, r2b, r5])
    else:
        tree.branch.set_last_revision_info(3, r5)
    tree.lock_read()
    self.addCleanup(tree.unlock)
    refs = self.make_refs(tree.branch)
    result = tree.branch.check(refs)
    ui.ui_factory = tests.TestUIFactory(stdout=BytesIO())
    result.report_results(True)
    self.assertContainsRe(b'revno does not match len', ui.ui_factory.stdout.getvalue())