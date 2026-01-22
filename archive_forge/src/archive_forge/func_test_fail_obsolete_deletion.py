from stat import S_ISDIR
from ... import controldir, errors, gpg, osutils, repository
from ... import revision as _mod_revision
from ... import tests, transport, ui
from ...tests import TestCaseWithTransport, TestNotApplicable, test_server
from ...transport import memory
from .. import inventory
from ..btree_index import BTreeGraphIndex
from ..groupcompress_repo import RepositoryFormat2a
from ..index import GraphIndex
from ..smart import client
def test_fail_obsolete_deletion(self):
    format = self.get_format()
    server = test_server.FakeNFSServer()
    self.start_server(server)
    t = transport.get_transport_from_url(server.get_url())
    bzrdir = self.get_format().initialize_on_transport(t)
    repo = bzrdir.create_repository()
    repo_transport = bzrdir.get_repository_transport(None)
    self.assertTrue(repo_transport.has('obsolete_packs'))
    repo_transport.put_bytes('obsolete_packs/.nfsblahblah', b'contents')
    repo._pack_collection._clear_obsolete_packs()
    self.assertTrue(repo_transport.has('obsolete_packs/.nfsblahblah'))