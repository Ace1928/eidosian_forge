from twisted.internet import defer, error, interfaces, reactor, task
from twisted.internet.main import installReactor
from twisted.internet.test.modulehelpers import NoReactor
from twisted.trial import unittest
from twisted.python import failure
def test_exitWithDefinedCode(self):
    """
        L{task.react} forwards the exit code specified by the C{SystemExit}
        error returned by the passed function, if any.
        """

    async def main(reactor):
        return await defer.fail(SystemExit(23))
    r = _FakeReactor()
    exitError = self.assertRaises(SystemExit, task.react, main, [], _reactor=r)
    self.assertEqual(23, exitError.code)