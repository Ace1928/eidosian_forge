import re
from breezy.bzr.tests.test_testament import (REV_1_SHORT, REV_1_SHORT_STRICT,
def test_testament_non_ascii(self):
    self.wt.commit('Non åssci message')
    long_out, err = self.run_bzr_raw('testament --long', encoding='utf-8')
    self.assertEqualDiff(err, b'')
    long_out, err = self.run_bzr_raw('testament --long', encoding='ascii')
    short_out, err = self.run_bzr_raw('testament', encoding='ascii')
    self.assertEqualDiff(err, b'')
    sha1_re = re.compile(b'sha1: (?P<sha1>[a-f0-9]+)$', re.M)
    sha1 = sha1_re.search(short_out).group('sha1')
    self.assertEqual(sha1, osutils.sha_string(long_out))