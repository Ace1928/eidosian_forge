from unittest import TestLoader
from .... import config, tests
from ....bzr.bzrdir import BzrDir
from ....tests import TestCaseInTempDir
from ..emailer import EmailSender
def test_diff_filename(self):
    sender, revid = self.get_sender()
    self.assertEqual('patch-1.diff', sender.diff_filename())