import os
from breezy import osutils
from breezy.bzr.testament import StrictTestament, StrictTestament3, Testament
from breezy.tests import TestCaseWithTransport
from breezy.tests.features import SymlinkFeature
def test_testament_with_contents(self):
    """Testament containing a file and a directory."""
    t = self.from_revision(self.b.repository, b'test@user-2')
    text_form = t.as_text()
    self.log('testament text form:\n%s' % text_form)
    self.assertEqualDiff(text_form, self.expected('rev_2'))
    actual_short = t.as_short_text()
    self.assertEqualDiff(actual_short, self.expected('rev_2_short'))