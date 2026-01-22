import os
import tempfile
from breezy import osutils, tests, transport, urlutils
def test_cwd_log(self):
    tmp_dir = osutils.realpath(tempfile.mkdtemp())
    self.permit_url('file:///')
    self.addCleanup(osutils.rmtree, tmp_dir)
    out, err = self.run_bzr('log', retcode=3, working_dir=tmp_dir)
    self.assertEqual('brz: ERROR: Not a branch: "%s/".\n' % (tmp_dir,), err)