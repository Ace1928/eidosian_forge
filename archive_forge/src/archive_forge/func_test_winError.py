import os
import socket
from unittest import skipIf
from twisted.internet.tcp import ECONNABORTED
from twisted.python.runtime import platform
from twisted.python.win32 import _ErrorFormatter, formatError
from twisted.trial.unittest import TestCase
def test_winError(self):
    """
        L{_ErrorFormatter.formatError} should return the message argument from
        the exception L{winError} returns, if L{winError} is supplied.
        """
    winCalls = []

    def winError(errorCode):
        winCalls.append(errorCode)
        return _MyWindowsException(errorCode, self.probeMessage)
    formatter = _ErrorFormatter(winError, lambda error: 'formatMessage: wrong message', {self.probeErrorCode: 'errorTab: wrong message'})
    message = formatter.formatError(self.probeErrorCode)
    self.assertEqual(message, self.probeMessage)