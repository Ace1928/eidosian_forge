import bz2
import tarfile
import zlib
from io import BytesIO
import fastbencode as bencode
from breezy import branch as _mod_branch
from breezy import controldir, errors, gpg, tests, transport, urlutils
from breezy.bzr import branch as _mod_bzrbranch
from breezy.bzr import inventory_delta, versionedfile
from breezy.bzr.smart import branch as smart_branch
from breezy.bzr.smart import bzrdir as smart_dir
from breezy.bzr.smart import packrepository as smart_packrepo
from breezy.bzr.smart import repository as smart_repo
from breezy.bzr.smart import request as smart_req
from breezy.bzr.smart import server, vfs
from breezy.bzr.testament import Testament
from breezy.tests import test_server
from breezy.transport import chroot, memory
def test_autopack_not_needed(self):
    tree = self.make_branch_and_tree('.', format='pack-0.92')
    repo = tree.branch.repository
    repo.lock_write()
    self.addCleanup(repo.unlock)
    for x in range(9):
        tree.commit('commit %s' % x)
    backing = self.get_transport()
    request = smart_packrepo.SmartServerPackRepositoryAutopack(backing)
    response = request.execute(b'')
    self.assertEqual(smart_req.SmartServerResponse((b'ok',)), response)
    repo._pack_collection.reload_pack_names()
    self.assertEqual(9, len(repo._pack_collection.names()))