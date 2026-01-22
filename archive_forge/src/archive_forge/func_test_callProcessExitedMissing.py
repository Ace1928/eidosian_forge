from twisted.internet._baseprocess import BaseProcess
from twisted.python.deprecate import getWarningMethod, setWarningMethod
from twisted.trial.unittest import TestCase
def test_callProcessExitedMissing(self):
    """
        L{BaseProcess._callProcessExited} emits a L{DeprecationWarning} if the
        object referred to by its C{proto} attribute has no C{processExited}
        method.
        """

    class FakeProto:
        pass
    reason = object()
    process = BaseProcess(FakeProto())
    self.addCleanup(setWarningMethod, getWarningMethod())
    warnings = []

    def collect(message, category, stacklevel):
        warnings.append((message, category, stacklevel))
    setWarningMethod(collect)
    process._callProcessExited(reason)
    [(message, category, stacklevel)] = warnings
    self.assertEqual(message, 'Since Twisted 8.2, IProcessProtocol.processExited is required.  %s.%s must implement it.' % (FakeProto.__module__, FakeProto.__name__))
    self.assertIs(category, DeprecationWarning)
    self.assertEqual(stacklevel, 0)