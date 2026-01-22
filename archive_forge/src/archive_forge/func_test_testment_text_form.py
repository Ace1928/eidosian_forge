import os
from breezy import osutils
from breezy.bzr.testament import StrictTestament, StrictTestament3, Testament
from breezy.tests import TestCaseWithTransport
from breezy.tests.features import SymlinkFeature
def test_testment_text_form(self):
    """Conversion of testament to canonical text form."""
    t = self.from_revision(self.b.repository, b'test@user-1')
    text_form = t.as_text()
    self.log('testament text form:\n%s' % text_form)
    self.assertEqualDiff(text_form, self.expected('rev_1'))
    short_text_form = t.as_short_text()
    self.assertEqualDiff(short_text_form, self.expected('rev_1_short'))