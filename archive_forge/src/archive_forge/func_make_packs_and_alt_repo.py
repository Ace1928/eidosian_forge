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
def make_packs_and_alt_repo(self, write_lock=False):
    """Create a pack repo with 3 packs, and access it via a second repo."""
    tree = self.make_branch_and_tree('.', format=self.get_format())
    tree.lock_write()
    self.addCleanup(tree.unlock)
    rev1 = tree.commit('one')
    rev2 = tree.commit('two')
    rev3 = tree.commit('three')
    r = repository.Repository.open('.')
    if write_lock:
        r.lock_write()
    else:
        r.lock_read()
    self.addCleanup(r.unlock)
    packs = r._pack_collection
    packs.ensure_loaded()
    return (tree, r, packs, [rev1, rev2, rev3])