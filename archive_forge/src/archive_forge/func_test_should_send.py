from unittest import TestLoader
from .... import config, tests
from ....bzr.bzrdir import BzrDir
from ....tests import TestCaseInTempDir
from ..emailer import EmailSender
def test_should_send(self):
    sender, revid = self.get_sender()
    self.assertEqual(True, sender.should_send())