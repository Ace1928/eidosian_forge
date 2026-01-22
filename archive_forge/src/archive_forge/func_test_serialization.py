import re
from .. import errors, gpg, mail_client, merge_directive, tests, trace
def test_serialization(self):
    time = 453
    timezone = 120
    md = self.make_merge_directive(b'example:', b'sha', time, timezone, 'http://example.com', patch=b'booga', patch_type='bundle')
    self.assertEqualDiff(self.OUTPUT1, b''.join(md.to_lines()))
    md = self.make_merge_directive(b'example:', b'sha', time, timezone, 'http://example.com', source_branch='http://example.org', patch=b'booga', patch_type='diff', message='Hi mom!')
    self.assertEqualDiff(self.OUTPUT2, b''.join(md.to_lines()))