from itertools import permutations
from dulwich.tests import TestCase
from ..diff_tree import (
from ..index import commit_tree
from ..object_store import MemoryObjectStore
from ..objects import Blob, ShaFile, Tree, TreeEntry
from .utils import F, ext_functest_builder, functest_builder, make_object
def test_similarity_score_cache(self):
    blob1 = make_object(Blob, data=b'ab\ncd\n')
    blob2 = make_object(Blob, data=b'ab\n')
    block_cache = {}
    self.assertEqual(50, _similarity_score(blob1, blob2, block_cache=block_cache))
    self.assertEqual({blob1.id, blob2.id}, set(block_cache))

    def fail_chunks():
        self.fail('Unexpected call to as_raw_chunks()')
    blob1.as_raw_chunks = blob2.as_raw_chunks = fail_chunks
    blob1.raw_length = lambda: 6
    blob2.raw_length = lambda: 3
    self.assertEqual(50, _similarity_score(blob1, blob2, block_cache=block_cache))