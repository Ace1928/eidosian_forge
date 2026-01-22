import os
from breezy import osutils, tests
from breezy.tests import features, script
from breezy.tests.scenarios import load_tests_apply_scenarios
def test_add_not_child(self):
    sr = script.ScriptRunner()
    self.make_branch_and_tree('tree1')
    self.make_branch_and_tree('tree2')
    self.build_tree(['tree1/a', 'tree2/b'])
    sr.run_script(self, '\n        $ brz add tree1/a tree2/b\n        2>brz: ERROR: Path "...tree2/b" is not a child of path "...tree1"\n        ')