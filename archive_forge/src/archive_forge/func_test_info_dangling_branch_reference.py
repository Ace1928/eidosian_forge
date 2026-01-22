import shutil
import sys
from breezy import (branch, controldir, errors, info, osutils, tests, upgrade,
from breezy.bzr import bzrdir
from breezy.transport import memory
def test_info_dangling_branch_reference(self):
    br = self.make_branch('target')
    br.create_checkout('from', lightweight=True)
    shutil.rmtree('target')
    out, err = self.run_bzr('info from')
    self.assertEqual(out, 'Dangling branch reference (format: 2a)\nLocation:\n   control directory: from\n  checkout of branch: target\n')
    self.assertEqual(err, '')