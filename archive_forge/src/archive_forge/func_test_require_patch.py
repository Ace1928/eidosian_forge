import re
from .. import errors, gpg, mail_client, merge_directive, tests, trace
def test_require_patch(self):
    time = 500.0
    timezone = 120
    self.assertRaises(errors.PatchMissing, merge_directive.MergeDirective, b'example:', b'sha', time, timezone, 'http://example.com', patch_type='bundle')
    md = merge_directive.MergeDirective(b'example:', b'sha1', time, timezone, 'http://example.com', source_branch='http://example.org', patch=b'', patch_type='diff')
    self.assertEqual(md.patch, b'')