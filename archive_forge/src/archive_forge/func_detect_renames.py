from itertools import permutations
from dulwich.tests import TestCase
from ..diff_tree import (
from ..index import commit_tree
from ..object_store import MemoryObjectStore
from ..objects import Blob, ShaFile, Tree, TreeEntry
from .utils import F, ext_functest_builder, functest_builder, make_object
def detect_renames(self, tree1, tree2, want_unchanged=False, **kwargs):
    detector = RenameDetector(self.store, **kwargs)
    return detector.changes_with_renames(tree1.id, tree2.id, want_unchanged=want_unchanged)