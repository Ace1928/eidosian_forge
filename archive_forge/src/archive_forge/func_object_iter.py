import os
import dulwich
from dulwich.repo import Repo as GitRepo
from ... import config, errors, revision
from ...repository import InterRepository, Repository
from .. import dir, repository, tests
from ..mapping import default_mapping
from ..object_store import BazaarObjectStore
from ..push import MissingObjectsIterator
def object_iter(self):
    store = BazaarObjectStore(self.bzr_tree.branch.repository, default_mapping)
    store_iterator = MissingObjectsIterator(store, self.bzr_tree.branch.repository)
    return (store, store_iterator)