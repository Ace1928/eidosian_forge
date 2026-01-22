import os
from breezy import osutils
from breezy.bzr.testament import StrictTestament, StrictTestament3, Testament
from breezy.tests import TestCaseWithTransport
from breezy.tests.features import SymlinkFeature
def test___init__(self):
    revision = self.b.repository.get_revision(b'test@user-2')
    tree = self.b.repository.revision_tree(b'test@user-2')
    testament_1 = self.testament_class()(revision, tree)
    text_1 = testament_1.as_short_text()
    text_2 = self.from_revision(self.b.repository, b'test@user-2').as_short_text()
    self.assertEqual(text_1, text_2)