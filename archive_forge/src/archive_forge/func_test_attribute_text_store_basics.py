from breezy import errors, gpg
from breezy import repository as _mod_repository
from breezy import revision as _mod_revision
from breezy import tests
from breezy.bzr import inventory, versionedfile, vf_repository
from breezy.bzr.tests.per_repository_vf import (
from breezy.tests.matchers import MatchesAncestry
from breezy.tests.scenarios import load_tests_apply_scenarios
def test_attribute_text_store_basics(self):
    """Test the basic behaviour of the text store."""
    tree = self.make_branch_and_tree('tree')
    repo = tree.branch.repository
    file_id = b'Foo:Bar'
    file_key = (file_id,)
    with tree.lock_write():
        self.assertEqual(set(), set(repo.texts.keys()))
        tree.add(['foo'], ['file'], [file_id])
        tree.put_file_bytes_non_atomic('foo', b'content\n')
        try:
            rev_key = (tree.commit('foo'),)
        except errors.IllegalPath:
            raise tests.TestNotApplicable('file_id %r cannot be stored on this platform for this repo format' % (file_id,))
        if repo._format.rich_root_data:
            root_commit = (tree.path2id(''),) + rev_key
            keys = {root_commit}
            parents = {root_commit: ()}
        else:
            keys = set()
            parents = {}
        keys.add(file_key + rev_key)
        parents[file_key + rev_key] = ()
        self.assertEqual(keys, set(repo.texts.keys()))
        self.assertEqual(parents, repo.texts.get_parent_map(repo.texts.keys()))
    tree2 = self.make_branch_and_tree('tree2')
    tree2.pull(tree.branch)
    tree2.put_file_bytes_non_atomic('foo', b'right\n')
    right_key = (tree2.commit('right'),)
    keys.add(file_key + right_key)
    parents[file_key + right_key] = (file_key + rev_key,)
    tree.put_file_bytes_non_atomic('foo', b'left\n')
    left_key = (tree.commit('left'),)
    keys.add(file_key + left_key)
    parents[file_key + left_key] = (file_key + rev_key,)
    tree.merge_from_branch(tree2.branch)
    tree.put_file_bytes_non_atomic('foo', b'merged\n')
    try:
        tree.auto_resolve()
    except errors.UnsupportedOperation:
        pass
    merge_key = (tree.commit('merged'),)
    keys.add(file_key + merge_key)
    parents[file_key + merge_key] = (file_key + left_key, file_key + right_key)
    repo.lock_read()
    self.addCleanup(repo.unlock)
    self.assertEqual(keys, set(repo.texts.keys()))
    self.assertEqual(parents, repo.texts.get_parent_map(repo.texts.keys()))