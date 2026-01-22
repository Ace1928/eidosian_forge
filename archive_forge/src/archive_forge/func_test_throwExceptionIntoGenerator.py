from io import StringIO
from twisted.internet import defer
from twisted.python import log
from twisted.python.reflect import qual
from twisted.spread import flavors, jelly, pb
from twisted.test.iosim import connectedServerAndClient
from twisted.trial import unittest
def test_throwExceptionIntoGenerator(self):
    """
        L{pb.CopiedFailure.throwExceptionIntoGenerator} will throw a
        L{RemoteError} into the given paused generator at the point where it
        last yielded.
        """
    original = pb.CopyableFailure(AttributeError('foo'))
    copy = jelly.unjelly(jelly.jelly(original, invoker=DummyInvoker()))
    exception = []

    def generatorFunc():
        try:
            yield None
        except pb.RemoteError as exc:
            exception.append(exc)
        else:
            self.fail('RemoteError not raised')
    gen = generatorFunc()
    gen.send(None)
    self.assertRaises(StopIteration, copy.throwExceptionIntoGenerator, gen)
    self.assertEqual(len(exception), 1)
    exc = exception[0]
    self.assertEqual(exc.remoteType, qual(AttributeError).encode('ascii'))
    self.assertEqual(exc.args, ('foo',))
    self.assertEqual(exc.remoteTraceback, 'Traceback unavailable\n')