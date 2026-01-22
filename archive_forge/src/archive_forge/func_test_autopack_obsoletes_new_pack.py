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
def test_autopack_obsoletes_new_pack(self):
    tree, r, packs, revs = self.make_packs_and_alt_repo(write_lock=True)
    packs._max_pack_count = lambda x: 1
    packs.pack_distribution = lambda x: [10]
    r.start_write_group()
    r.revisions.insert_record_stream([versionedfile.FulltextContentFactory((b'bogus-rev',), (), None, b'bogus-content\n')])
    r.commit_write_group()
    names = packs.names()
    self.assertEqual(1, len(names))
    self.assertEqual([names[0] + '.pack'], packs._pack_transport.list_dir('.'))