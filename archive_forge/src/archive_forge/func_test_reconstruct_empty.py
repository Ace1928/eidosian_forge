from unittest import TestCase
import patiencediff
from .. import multiparent, tests
def test_reconstruct_empty(self):
    vf = multiparent.MultiMemoryVersionedFile()
    vf.add_version([], b'a', [])
    self.assertEqual([], self.reconstruct_version(vf, b'a'))