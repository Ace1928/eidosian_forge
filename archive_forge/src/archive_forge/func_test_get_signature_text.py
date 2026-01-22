import os
import dulwich
from dulwich.repo import Repo as GitRepo
from ... import config, errors, revision
from ...repository import InterRepository, Repository
from .. import dir, repository, tests
from ..mapping import default_mapping
from ..object_store import BazaarObjectStore
from ..push import MissingObjectsIterator
def test_get_signature_text(self):
    self.assertRaises(errors.NoSuchRevision, self.git_repo.get_signature_text, revision.NULL_REVISION)