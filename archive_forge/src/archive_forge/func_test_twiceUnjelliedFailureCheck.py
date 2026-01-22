from io import StringIO
from twisted.internet import defer
from twisted.python import log
from twisted.python.reflect import qual
from twisted.spread import flavors, jelly, pb
from twisted.test.iosim import connectedServerAndClient
from twisted.trial import unittest
def test_twiceUnjelliedFailureCheck(self):
    """
        The object which results from jellying a L{CopyableFailure}, unjellying
        the result, creating a new L{CopyableFailure} from the result of that,
        jellying it, and finally unjellying the result of that has a check
        method which behaves the same way as the original L{CopyableFailure}'s
        check method.
        """
    original = pb.CopyableFailure(ZeroDivisionError())
    self.assertIs(original.check(ZeroDivisionError), ZeroDivisionError)
    self.assertIs(original.check(ArithmeticError), ArithmeticError)
    copiedOnce = jelly.unjelly(jelly.jelly(original, invoker=DummyInvoker()))
    derivative = pb.CopyableFailure(copiedOnce)
    copiedTwice = jelly.unjelly(jelly.jelly(derivative, invoker=DummyInvoker()))
    self.assertIs(copiedTwice.check(ZeroDivisionError), ZeroDivisionError)
    self.assertIs(copiedTwice.check(ArithmeticError), ArithmeticError)