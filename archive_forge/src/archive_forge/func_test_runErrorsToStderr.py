import os
import sys
from io import StringIO
from unittest import skipIf
from twisted.copyright import version
from twisted.internet.defer import Deferred
from twisted.internet.testing import MemoryReactor
from twisted.mail import smtp
from twisted.mail.scripts import mailmail
from twisted.mail.scripts.mailmail import parseOptions
from twisted.python.failure import Failure
from twisted.python.runtime import platformType
from twisted.trial.unittest import TestCase
@skipIf(platformType == 'win32', 'mailmail.run() does not work on win32 due to lack of support for getuid()')
def test_runErrorsToStderr(self):
    """
        Call L{mailmail.run}, and specify I{-oep} to print errors
        to stderr.  The sender, to, and printErrors options should be
        set and there should be no failure.
        """
    argv = ('test_mailmail.py', 'invaliduser2@example.com', '-oep')
    stdin = StringIO('\n')
    self.patch(sys, 'argv', argv)
    self.patch(sys, 'stdin', stdin)
    mailmail.run()
    self.assertEqual(self.options.sender, mailmail.getlogin())
    self.assertEqual(self.options.to, ['invaliduser2@example.com'])
    self.assertTrue(self.options.printErrors)
    self.assertIsNone(mailmail.failed)