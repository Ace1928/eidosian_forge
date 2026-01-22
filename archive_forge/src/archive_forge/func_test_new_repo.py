from breezy import revision as _mod_revision
from breezy.bzr.inventorytree import InventoryTreeChange
from breezy.tests.per_repository import TestCaseWithRepository
def test_new_repo(self):
    branch = self.make_branch('foo')
    branch.lock_write()
    self.addCleanup(branch.unlock)
    self.overrideEnv('BRZ_EMAIL', 'foo@sample.com')
    builder = branch.get_commit_builder([], branch.get_config_stack())
    list(builder.record_iter_changes(None, _mod_revision.NULL_REVISION, [InventoryTreeChange(b'TREE_ROOT', (None, ''), True, (False, True), (None, None), (None, ''), (None, 'directory'), (None, False))]))
    builder.finish_inventory()
    rev_id = builder.commit('first post')
    result = branch.repository.check(None, check_repo=True)
    result.report_results(True)
    log = self.get_log()
    self.assertFalse('Missing' in log, 'Something was missing in %r' % log)