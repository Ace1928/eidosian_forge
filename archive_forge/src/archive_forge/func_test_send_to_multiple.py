from unittest import TestLoader
from .... import config, tests
from ....bzr.bzrdir import BzrDir
from ....tests import TestCaseInTempDir
from ..emailer import EmailSender
def test_send_to_multiple(self):
    sender, revid = self.get_sender(multiple_to_configured_config)
    self.assertEqual(['Sample <foo@example.com>', 'Other <baz@bar.com>'], sender.to())
    self.assertEqual(['Sample <foo@example.com>', 'Other <baz@bar.com>'], sender._command_line()[-2:])