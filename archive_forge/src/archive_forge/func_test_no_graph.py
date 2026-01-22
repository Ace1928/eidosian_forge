from .. import annotate, errors, revision, tests
from ..bzr import knit
def test_no_graph(self):
    self.make_no_graph_texts()
    self.assertAnnotateEqual([(self.fa_key,), (self.fa_key,)], self.fa_key)
    self.assertAnnotateEqual([(self.fb_key,), (self.fb_key,)], self.fb_key)