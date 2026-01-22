import os
import dulwich
from dulwich.repo import Repo as GitRepo
from ... import config, errors, revision
from ...repository import InterRepository, Repository
from .. import dir, repository, tests
from ..mapping import default_mapping
from ..object_store import BazaarObjectStore
from ..push import MissingObjectsIterator
def test_get_known_graph_ancestry(self):
    cid = self._do_commit()
    revid = default_mapping.revision_id_foreign_to_bzr(cid)
    g = self.git_repo.get_known_graph_ancestry([revid])
    self.assertEqual(frozenset([revid]), g.heads([revid]))
    self.assertEqual([(revid, 0, (1,), True)], [(n.key, n.merge_depth, n.revno, n.end_of_merge) for n in g.merge_sort(revid)])