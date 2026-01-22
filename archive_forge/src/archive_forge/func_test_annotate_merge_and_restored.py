from .. import annotate, errors, revision, tests
from ..bzr import knit
def test_annotate_merge_and_restored(self):
    self.make_merge_and_restored_text()
    self.assertAnnotateEqual([(self.fa_key,), (self.fa_key, self.fc_key)], self.fd_key)