from .. import annotate, errors, revision, tests
from ..bzr import knit
def test_needed_keys_with_special_text(self):
    self.make_many_way_common_merge_text()
    spec_key = (b'f-id', revision.CURRENT_REVISION)
    spec_text = b'simple\nnew content\nlocally modified\n'
    self.ann.add_special_text(spec_key, [self.fd_key, self.fe_key], spec_text)
    keys, ann_keys = self.ann._get_needed_keys(spec_key)
    self.assertEqual([self.fa_key, self.fb_key, self.fc_key, self.fd_key, self.fe_key], sorted(keys))
    self.assertEqual([spec_key], sorted(ann_keys))