from breezy import errors, tests, urlutils
from breezy.bzr import remote
from breezy.tests.per_repository import TestCaseWithRepository
def test_merge_commit(self):
    base_tree, stacked_tree = self.make_stacked_target()
    self.build_tree_contents([('base/f1.txt', b'new content\n')])
    r3_key = (b'rev3-id',)
    base_tree.commit('second base', rev_id=r3_key[0])
    to_be_merged_tree = base_tree.controldir.sprout('merged').open_workingtree()
    self.build_tree(['merged/f2.txt'])
    to_be_merged_tree.add(['f2.txt'], ids=[b'f2.txt-id'])
    to_merge_key = (b'to-merge-rev-id',)
    to_be_merged_tree.commit('new-to-be-merged', rev_id=to_merge_key[0])
    stacked_tree.merge_from_branch(to_be_merged_tree.branch)
    merged_key = (b'merged-rev-id',)
    stacked_tree.commit('merge', rev_id=merged_key[0])
    stacked_only_repo = self.get_only_repo(stacked_tree)
    all_keys = [self.r1_key, self.r2_key, r3_key, to_merge_key, merged_key]
    self.assertPresent([to_merge_key, merged_key], stacked_only_repo.revisions, all_keys)
    self.assertPresent([self.r2_key, r3_key, to_merge_key, merged_key], stacked_only_repo.inventories, all_keys)