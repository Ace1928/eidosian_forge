from .. import annotate, errors, revision, tests
from ..bzr import knit
def test_annotate_many_way_common_merge_text(self):
    self.make_many_way_common_merge_text()
    self.assertAnnotateEqual([(self.fa_key,), (self.fb_key, self.fc_key, self.fe_key)], self.ff_key)