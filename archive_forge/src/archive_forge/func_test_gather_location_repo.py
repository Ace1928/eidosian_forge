import sys
from .. import branch as _mod_branch
from .. import controldir, errors, info
from .. import repository as _mod_repository
from .. import tests, workingtree
from ..bzr import branch as _mod_bzrbranch
def test_gather_location_repo(self):
    srepo = self.make_repository('shared', shared=True)
    self.assertEqual([('shared repository', srepo.controldir.root_transport.base)], info.gather_location_info(srepo, control=srepo.controldir))
    urepo = self.make_repository('unshared')
    self.assertEqual([('repository', urepo.controldir.root_transport.base)], info.gather_location_info(urepo, control=urepo.controldir))