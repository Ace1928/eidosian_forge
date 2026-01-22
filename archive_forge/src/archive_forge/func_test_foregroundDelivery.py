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
def test_foregroundDelivery(self):
    """
        The I{-odf} flags specifies foreground delivery.
        """
    stdin = StringIO('\n')
    self.patch(sys, 'stdin', stdin)
    o = parseOptions('-odf')
    self.assertFalse(o.background)