from breezy import config, tests
from breezy.tests import features
def test_add_alias_outputs_nothing(self):
    out, err = self.run_bzr('alias commit="commit --strict"')
    self.assertEqual('', out)