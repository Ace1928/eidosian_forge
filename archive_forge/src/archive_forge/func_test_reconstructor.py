from unittest import TestCase
import patiencediff
from .. import multiparent, tests
def test_reconstructor(self):
    vf = self.make_vf()
    self.assertEqual([b'a\n', b'b\n'], self.reconstruct(vf, b'rev-a', 0, 2))
    self.assertEqual([b'c\n', b'd\n'], self.reconstruct(vf, b'rev-a', 2, 4))
    self.assertEqual([b'e\n', b'f\n'], self.reconstruct(vf, b'rev-c', 2, 4))
    self.assertEqual([b'a\n', b'b\n', b'e\n', b'f\n'], self.reconstruct(vf, b'rev-c', 0, 4))
    self.assertEqual([b'a\n', b'b\n', b'e\n', b'f\n'], self.reconstruct_version(vf, b'rev-c'))