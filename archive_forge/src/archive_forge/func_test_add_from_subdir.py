import os
from breezy import osutils, tests
from breezy.tests import features, script
from breezy.tests.scenarios import load_tests_apply_scenarios
def test_add_from_subdir(self):
    base_tree = self.make_branch_and_tree('base')
    self.build_tree(['base/a', 'base/b/', 'base/b/c', 'base/b/d'])
    base_tree.add(['a', 'b', 'b/c', 'b/d'])
    base_tree.commit('foo')
    new_tree = self.make_branch_and_tree('new')
    self.build_tree(['new/c', 'new/d'])
    out, err = self.run_bzr('add --file-ids-from ../base/b', working_dir='new')
    self.assertEqual('', err)
    self.assertEqualDiff('adding c w/ file id from b/c\nadding d w/ file id from b/d\n', out)
    new_tree = new_tree.controldir.open_workingtree('new')
    self.assertEqual(base_tree.path2id('b/c'), new_tree.path2id('c'))
    self.assertEqual(base_tree.path2id('b/d'), new_tree.path2id('d'))