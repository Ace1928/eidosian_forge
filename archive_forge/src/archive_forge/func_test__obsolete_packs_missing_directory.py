from stat import S_ISDIR
import breezy
from breezy import controldir, errors, osutils, repository
from breezy import revision as _mod_revision
from breezy import tests, transport, upgrade, workingtree
from breezy.bzr import (btree_index, bzrdir, groupcompress_repo, inventory,
from breezy.bzr import repository as bzrrepository
from breezy.bzr import versionedfile, vf_repository, vf_search
from breezy.bzr.btree_index import BTreeBuilder, BTreeGraphIndex
from breezy.bzr.index import GraphIndex
from breezy.errors import UnknownFormatError
from breezy.repository import RepositoryFormat
from breezy.tests import TestCase, TestCaseWithTransport
def test__obsolete_packs_missing_directory(self):
    tree, r, packs, revs = self.make_packs_and_alt_repo(write_lock=True)
    r.control_transport.rmdir('obsolete_packs')
    names = packs.names()
    pack = packs.get_pack_by_name(names[0])
    packs._remove_pack_from_memory(pack)
    packs._obsolete_packs([pack])
    self.assertEqual([n + '.pack' for n in names[1:]], sorted(packs._pack_transport.list_dir('.')))
    self.assertEqual(names[1:], sorted({osutils.splitext(n)[0] for n in packs._index_transport.list_dir('.')}))