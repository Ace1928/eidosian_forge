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
def test_adding_pack_does_not_record_pack_names_from_other_repositories(self):
    base = self.make_branch_and_tree('base', format=self.get_format())
    base.commit('foo')
    referencing = self.make_branch_and_tree('repo', format=self.get_format())
    referencing.branch.repository.add_fallback_repository(base.branch.repository)
    local_tree = referencing.branch.create_checkout('local')
    local_tree.commit('bar')
    new_instance = referencing.controldir.open_repository()
    new_instance.lock_read()
    self.addCleanup(new_instance.unlock)
    new_instance._pack_collection.ensure_loaded()
    self.assertEqual(1, len(new_instance._pack_collection.all_packs()))