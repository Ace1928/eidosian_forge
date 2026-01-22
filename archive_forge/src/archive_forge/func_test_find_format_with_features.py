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
def test_find_format_with_features(self):
    tree = self.make_branch_and_tree('.', format='2a')
    tree.branch.repository.update_feature_flags({b'name': b'necessity'})
    found_format = bzrrepository.RepositoryFormatMetaDir.find_format(tree.controldir)
    self.assertIsInstance(found_format, bzrrepository.RepositoryFormatMetaDir)
    self.assertEqual(found_format.features.get(b'name'), b'necessity')
    self.assertRaises(bzrdir.MissingFeature, found_format.check_support_status, True)
    self.addCleanup(bzrrepository.RepositoryFormatMetaDir.unregister_feature, b'name')
    bzrrepository.RepositoryFormatMetaDir.register_feature(b'name')
    found_format.check_support_status(True)