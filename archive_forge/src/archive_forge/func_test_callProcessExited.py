from twisted.internet._baseprocess import BaseProcess
from twisted.python.deprecate import getWarningMethod, setWarningMethod
from twisted.trial.unittest import TestCase
def test_callProcessExited(self):
    """
        L{BaseProcess._callProcessExited} calls the C{processExited} method of
        its C{proto} attribute and passes it a L{Failure} wrapping the given
        exception.
        """

    class FakeProto:
        reason = None

        def processExited(self, reason):
            self.reason = reason
    reason = RuntimeError('fake reason')
    process = BaseProcess(FakeProto())
    process._callProcessExited(reason)
    process.proto.reason.trap(RuntimeError)
    self.assertIs(reason, process.proto.reason.value)