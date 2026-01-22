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
def test_concurrent_pack_during_autopack(self):
    tree = self.make_branch_and_tree('tree')
    with tree.lock_write():
        for i in range(9):
            tree.commit('rev %d' % (i,))
        r2 = repository.Repository.open('tree')
        with r2.lock_write():
            autopack_count = [0]
            r1 = tree.branch.repository
            orig = r1._pack_collection.pack_distribution

            def trigger_during_auto(*args, **kwargs):
                ret = orig(*args, **kwargs)
                if not autopack_count[0]:
                    r2.pack()
                autopack_count[0] += 1
                return ret
            r1._pack_collection.pack_distribution = trigger_during_auto
            tree.commit('autopack-rev')
            self.assertEqual([2], autopack_count)