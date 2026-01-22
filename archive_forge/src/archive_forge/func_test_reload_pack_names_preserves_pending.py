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
def test_reload_pack_names_preserves_pending(self):
    tree, r, packs, revs = self.make_packs_and_alt_repo(write_lock=True)
    orig_names = packs.names()
    orig_at_load = packs._packs_at_load
    to_remove_name = next(iter(orig_names))
    r.start_write_group()
    self.addCleanup(r.abort_write_group)
    r.texts.insert_record_stream([versionedfile.FulltextContentFactory((b'text', b'rev'), (), None, b'content\n')])
    new_pack = packs._new_pack
    self.assertTrue(new_pack.data_inserted())
    new_pack.finish()
    packs.allocate(new_pack)
    packs._new_pack = None
    removed_pack = packs.get_pack_by_name(to_remove_name)
    packs._remove_pack_from_memory(removed_pack)
    names = packs.names()
    all_nodes, deleted_nodes, new_nodes, _ = packs._diff_pack_names()
    new_names = {x[0] for x in new_nodes}
    self.assertEqual(names, sorted([x[0] for x in all_nodes]))
    self.assertEqual(set(names) - set(orig_names), new_names)
    self.assertEqual({new_pack.name}, new_names)
    self.assertEqual([to_remove_name], sorted([x[0] for x in deleted_nodes]))
    packs.reload_pack_names()
    reloaded_names = packs.names()
    self.assertEqual(orig_at_load, packs._packs_at_load)
    self.assertEqual(names, reloaded_names)
    all_nodes, deleted_nodes, new_nodes, _ = packs._diff_pack_names()
    new_names = {x[0] for x in new_nodes}
    self.assertEqual(names, sorted([x[0] for x in all_nodes]))
    self.assertEqual(set(names) - set(orig_names), new_names)
    self.assertEqual({new_pack.name}, new_names)
    self.assertEqual([to_remove_name], sorted([x[0] for x in deleted_nodes]))