from unittest import TestLoader
from .... import config, tests
from ....bzr.bzrdir import BzrDir
from ....tests import TestCaseInTempDir
from ..emailer import EmailSender
def test_command_line(self):
    sender, revid = self.get_sender()
    self.assertEqual(['mail', '-s', sender.subject(), '-a', 'From: ' + sender.from_address()] + sender.to(), sender._command_line())