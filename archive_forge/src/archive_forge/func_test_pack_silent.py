import os
from breezy import tests
def test_pack_silent(self):
    """pack command has no intrinsic output."""
    self.make_branch('.')
    out, err = self.run_bzr('pack')
    self.assertEqual('', out)
    self.assertEqual('', err)