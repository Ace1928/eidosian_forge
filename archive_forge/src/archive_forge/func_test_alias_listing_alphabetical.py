from breezy import config, tests
from breezy.tests import features
def test_alias_listing_alphabetical(self):
    self.run_bzr('alias commit="commit --strict"')
    self.run_bzr('alias ll="log --short"')
    self.run_bzr('alias add="add -q"')
    out, err = self.run_bzr('alias')
    self.assertEqual('brz alias add="add -q"\nbrz alias commit="commit --strict"\nbrz alias ll="log --short"\n', out)