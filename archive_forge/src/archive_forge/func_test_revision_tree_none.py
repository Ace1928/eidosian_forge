import os
import dulwich
from dulwich.repo import Repo as GitRepo
from ... import config, errors, revision
from ...repository import InterRepository, Repository
from .. import dir, repository, tests
from ..mapping import default_mapping
from ..object_store import BazaarObjectStore
from ..push import MissingObjectsIterator
def test_revision_tree_none(self):
    repo = self.git_repo
    tree = repo.revision_tree(revision.NULL_REVISION)
    self.assertEqual(tree.get_revision_id(), revision.NULL_REVISION)