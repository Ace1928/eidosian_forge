from .. import annotate, errors, revision, tests
from ..bzr import knit
def make_merge_and_restored_text(self):
    self.make_simple_text()
    self.vf.add_lines(self.fc_key, [self.fb_key], [b'simple\n', b'content\n'])
    self.vf.add_lines(self.fd_key, [self.fa_key, self.fc_key], [b'simple\n', b'content\n'])