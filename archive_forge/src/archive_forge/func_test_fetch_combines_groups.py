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
def test_fetch_combines_groups(self):
    builder = self.make_branch_builder('source', format='2a')
    builder.start_series()
    builder.build_snapshot(None, [('add', ('', b'root-id', 'directory', '')), ('add', ('file', b'file-id', 'file', b'content\n'))], revision_id=b'1')
    builder.build_snapshot([b'1'], [('modify', ('file', b'content-2\n'))], revision_id=b'2')
    builder.finish_series()
    source = builder.get_branch()
    target = self.make_repository('target', format='2a')
    target.fetch(source.repository)
    target.lock_read()
    self.addCleanup(target.unlock)
    details = target.texts._index.get_build_details([(b'file-id', b'1'), (b'file-id', b'2')])
    file_1_details = details[b'file-id', b'1']
    file_2_details = details[b'file-id', b'2']
    self.assertEqual(file_1_details[0][:3], file_2_details[0][:3])