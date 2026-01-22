from .. import annotate, errors, revision, tests
from ..bzr import knit
def test_annotate_flat_simple(self):
    self.make_simple_text()
    self.assertEqual([(self.fa_key, b'simple\n'), (self.fa_key, b'content\n')], self.ann.annotate_flat(self.fa_key))
    self.assertEqual([(self.fa_key, b'simple\n'), (self.fb_key, b'new content\n')], self.ann.annotate_flat(self.fb_key))