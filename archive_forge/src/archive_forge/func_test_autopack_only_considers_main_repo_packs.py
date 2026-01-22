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
def test_autopack_only_considers_main_repo_packs(self):
    format = self.get_format()
    base = self.make_branch_and_tree('base', format=format)
    base.commit('foo')
    tree = self.make_branch_and_tree('repo', format=format)
    tree.branch.repository.add_fallback_repository(base.branch.repository)
    trans = tree.branch.repository.controldir.get_repository_transport(None)
    local_tree = tree.branch.create_checkout('local')
    for x in range(9):
        local_tree.commit('commit %s' % x)
    index = self.index_class(trans, 'pack-names', None)
    self.assertEqual(9, len(list(index.iter_all_entries())))
    local_tree.commit('commit triggering pack')
    index = self.index_class(trans, 'pack-names', None)
    self.assertEqual(1, len(list(index.iter_all_entries())))
    tree = tree.controldir.open_workingtree()
    check_result = tree.branch.repository.check([tree.branch.last_revision()])
    nb_files = 5
    if tree.branch.repository._format.supports_chks:
        nb_files += 1
    obsolete_files = list(trans.list_dir('obsolete_packs'))
    self.assertFalse('foo' in obsolete_files)
    self.assertFalse('bar' in obsolete_files)
    self.assertEqual(10 * nb_files, len(obsolete_files))
    large_pack_name = list(index.iter_all_entries())[0][1][0]
    local_tree.commit('commit not triggering pack')
    index = self.index_class(trans, 'pack-names', None)
    self.assertEqual(2, len(list(index.iter_all_entries())))
    pack_names = [node[1][0] for node in index.iter_all_entries()]
    self.assertTrue(large_pack_name in pack_names)