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
def test_setFrom(self):
    """
        When a message has no I{From:} header, a I{From:} value can be
        specified with the I{-F} flag.
        """
    stdin = StringIO('To: invaliduser2@example.com\nSubject: A wise guy?\n\n')
    self.patch(sys, 'stdin', stdin)
    o = parseOptions(['-F', 'Larry <invaliduser1@example.com>', '-t'])
    self.assertEqual(o.sender, 'Larry <invaliduser1@example.com>')