import os
from dulwich.repo import Repo as GitRepo
from ...controldir import ControlDir
from ...tests.blackbox import ExternalBase
from ...tests.features import PluginLoadedFeature
from ...tests.script import TestCaseWithTransportAndScript
from ...workingtree import WorkingTree
from .. import tests
def test_branch_with_nested_trees(self):
    orig = self.make_branch_and_tree('source', format='git')
    subtree = self.make_branch_and_tree('source/subtree', format='git')
    self.build_tree(['source/subtree/a'])
    self.build_tree_contents([('source/.gitmodules', '[submodule "subtree"]\n    path = subtree\n    url = %s\n' % subtree.user_url)])
    subtree.add(['a'])
    subtree.commit('add subtree contents')
    orig.add_reference(subtree)
    orig.add(['.gitmodules'])
    orig.commit('add subtree')
    self.run_bzr('branch source target')
    target = WorkingTree.open('target')
    target_subtree = WorkingTree.open('target/subtree')
    self.assertTreesEqual(orig, target)
    self.assertTreesEqual(subtree, target_subtree)