from .. import annotate, errors, revision, tests
from ..bzr import knit
def test_record_annotation_removes_texts(self):
    self.make_many_way_common_merge_text()
    for x in self.ann._get_needed_texts(self.ff_key):
        continue
    self.assertEqual({self.fa_key: 3, self.fb_key: 1, self.fc_key: 1, self.fd_key: 1, self.fe_key: 1, self.ff_key: 1}, self.ann._num_needed_children)
    self.assertEqual([self.fa_key, self.fb_key, self.fc_key, self.fd_key, self.fe_key, self.ff_key], sorted(self.ann._text_cache.keys()))
    self.ann._record_annotation(self.fa_key, [], [])
    self.ann._record_annotation(self.fb_key, [self.fa_key], [])
    self.assertEqual({self.fa_key: 2, self.fb_key: 1, self.fc_key: 1, self.fd_key: 1, self.fe_key: 1, self.ff_key: 1}, self.ann._num_needed_children)
    self.assertTrue(self.fa_key in self.ann._text_cache)
    self.assertTrue(self.fa_key in self.ann._annotations_cache)
    self.ann._record_annotation(self.fc_key, [self.fa_key], [])
    self.ann._record_annotation(self.fd_key, [self.fb_key, self.fc_key], [])
    self.assertEqual({self.fa_key: 1, self.fb_key: 0, self.fc_key: 0, self.fd_key: 1, self.fe_key: 1, self.ff_key: 1}, self.ann._num_needed_children)
    self.assertTrue(self.fa_key in self.ann._text_cache)
    self.assertTrue(self.fa_key in self.ann._annotations_cache)
    self.assertFalse(self.fb_key in self.ann._text_cache)
    self.assertFalse(self.fb_key in self.ann._annotations_cache)
    self.assertFalse(self.fc_key in self.ann._text_cache)
    self.assertFalse(self.fc_key in self.ann._annotations_cache)