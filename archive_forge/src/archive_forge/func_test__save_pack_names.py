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
def test__save_pack_names(self):
    tree, r, packs, revs = self.make_packs_and_alt_repo(write_lock=True)
    names = packs.names()
    pack = packs.get_pack_by_name(names[0])
    packs._remove_pack_from_memory(pack)
    packs._save_pack_names(obsolete_packs=[pack])
    cur_packs = packs._pack_transport.list_dir('.')
    self.assertEqual([n + '.pack' for n in names[1:]], sorted(cur_packs))
    obsolete_packs = packs.transport.list_dir('obsolete_packs')
    obsolete_names = {osutils.splitext(n)[0] for n in obsolete_packs}
    self.assertEqual([pack.name], sorted(obsolete_names))