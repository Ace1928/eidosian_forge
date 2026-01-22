from breezy import config, tests
from breezy.tests import features
def test_remove_known_alias(self):
    self.run_bzr('alias commit="commit --strict"')
    out, err = self.run_bzr('alias commit')
    self.assertEqual('brz alias commit="commit --strict"\n', out)
    out, err = self.run_bzr('alias --remove commit')
    self.assertEqual('', out)
    out, err = self.run_bzr('alias commit')
    self.assertEqual('brz alias: commit: not found\n', out)