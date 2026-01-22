from .. import tests
from . import features
def test_newlines(self):
    self.assertChunksToLines([b'\n'], [b'\n'], alreadly_lines=True)
    self.assertChunksToLines([b'\n'], [b'', b'\n', b''])
    self.assertChunksToLines([b'\n'], [b'\n', b''])
    self.assertChunksToLines([b'\n'], [b'', b'\n'])
    self.assertChunksToLines([b'\n', b'\n', b'\n'], [b'\n\n\n'])
    self.assertChunksToLines([b'\n', b'\n', b'\n'], [b'\n', b'\n', b'\n'], alreadly_lines=True)