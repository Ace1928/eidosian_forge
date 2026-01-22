from breezy import errors, gpg
from breezy import repository as _mod_repository
from breezy import revision as _mod_revision
from breezy import tests
from breezy.bzr import inventory, versionedfile, vf_repository
from breezy.bzr.tests.per_repository_vf import (
from breezy.tests.matchers import MatchesAncestry
from breezy.tests.scenarios import load_tests_apply_scenarios
def test_item_keys_introduced_by(self):
    tree = self.make_branch_and_tree('t')
    self.build_tree(['t/foo'])
    tree.add('foo', ids=b'file1')
    tree.commit('message', rev_id=b'rev_id')
    repo = tree.branch.repository
    repo.lock_write()
    repo.start_write_group()
    try:
        repo.sign_revision(b'rev_id', gpg.LoopbackGPGStrategy(None))
    except errors.UnsupportedOperation:
        signature_texts = []
    else:
        signature_texts = [b'rev_id']
    repo.commit_write_group()
    repo.unlock()
    repo.lock_read()
    self.addCleanup(repo.unlock)
    expected_item_keys = [('file', b'file1', [b'rev_id']), ('inventory', None, [b'rev_id']), ('signatures', None, signature_texts), ('revisions', None, [b'rev_id'])]
    item_keys = list(repo.item_keys_introduced_by([b'rev_id']))
    item_keys = [(kind, file_id, list(versions)) for kind, file_id, versions in item_keys]
    if repo.supports_rich_root():
        inv = repo.get_inventory(b'rev_id')
        root_item_key = ('file', inv.root.file_id, [b'rev_id'])
        self.assertIn(root_item_key, item_keys)
        item_keys.remove(root_item_key)
    self.assertEqual(expected_item_keys, item_keys)