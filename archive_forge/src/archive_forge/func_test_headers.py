from unittest import TestLoader
from .... import config, tests
from ....bzr.bzrdir import BzrDir
from ....tests import TestCaseInTempDir
from ..emailer import EmailSender
def test_headers(self):
    sender, revid = self.get_sender()
    self.assertEqual({'X-Cheese': 'to the rescue!'}, sender.extra_headers())