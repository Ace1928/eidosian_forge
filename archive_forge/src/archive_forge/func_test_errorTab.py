import os
import socket
from unittest import skipIf
from twisted.internet.tcp import ECONNABORTED
from twisted.python.runtime import platform
from twisted.python.win32 import _ErrorFormatter, formatError
from twisted.trial.unittest import TestCase
def test_errorTab(self):
    """
        L{_ErrorFormatter.formatError} should use C{errorTab} if it is supplied
        and contains the requested error code.
        """
    formatter = _ErrorFormatter(None, None, {self.probeErrorCode: self.probeMessage})
    message = formatter.formatError(self.probeErrorCode)
    self.assertEqual(message, self.probeMessage)